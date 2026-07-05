from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import logging
import docker
from pathlib import Path
import threading
from swebench.harness.docker_build import (
    build_base_images,
    build_env_images,
    build_container,
    cleanup_container,
    remove_image,
)
from swebench.harness.utils import (
    load_swebench_dataset,
)
from swebench.harness.test_spec.test_spec import make_test_spec
import re
import time


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def compact_text(text, max_lines=40, max_chars=8000):
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = ["... output truncated ...", *lines[-max_lines:]]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "... output truncated ...\n" + text[-max_chars:]
    return text


class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, instance_id, status, detail=""):
        with self.lock:
            self.done += 1
            if status == "ok":
                self.ok += 1
            elif status == "skipped":
                self.skipped += 1
            else:
                self.failed += 1

            elapsed = time.time() - self.start_time
            rate = self.done / elapsed if elapsed > 0 else 0
            remaining = self.total - self.done
            eta = format_duration(remaining / rate) if rate > 0 and remaining else "0s"
            suffix = f" {detail}" if detail else ""
            print(
                f"[call-hier] {self.done}/{self.total} "
                f"ok={self.ok} skipped={self.skipped} failed={self.failed} "
                f"elapsed={format_duration(elapsed)} eta={eta} "
                f"last={instance_id} status={status}{suffix}",
                flush=True,
            )


def sh_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def should_include(instance_id, instance_ids):
    return not instance_ids or instance_id in instance_ids


def parse_precomputed_bm25_files(instance):
    code_blocks = re.findall(r"\[end of (?i:readme)(?:[^\]\n]*)?\]\n(.*?)\n<\/code>", instance["text"], re.DOTALL)
    if not code_blocks:
        return []
    return re.findall(r"\n\[start of (.*?)\]\n", code_blocks[0])


def oracle_files_from_patch(instance):
    try:
        from swebench.harness.utils import get_modified_files

        return get_modified_files(instance.get("patch", ""))
    except Exception:
        logger = logging.getLogger("swebench.harness.run_call_hier")
        logger.exception("Failed to parse oracle files from patch")
        return []


def write_retrieval_inputs(output_dir, instance):
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "problem_statement.txt"
    patch_path = output_dir / "gold_patch.diff"
    query_path.write_text(instance.get("problem_statement", ""), encoding="utf-8")
    patch_path.write_text(instance.get("patch", ""), encoding="utf-8")
    return query_path, patch_path


def run_container_retrieval(
    container,
    instance,
    output_dir,
    retrieval_source,
    retrieval_level,
    retrieval_k,
    include_tests,
    logger,
    verbose_lsp_output=False,
):
    query_path, patch_path = write_retrieval_inputs(output_dir, instance)
    include_tests_arg = "--include-tests" if include_tests else ""
    command = (
        "set -e; "
        "/opt/miniconda3/envs/testbed/bin/python /headless_lsp/retrieve_context.py "
        "--workspace /testbed "
        f"--mode {sh_quote(retrieval_source)} "
        f"--level {sh_quote(retrieval_level)} "
        f"--k {int(retrieval_k)} "
        f"--query-file /output/{sh_quote(query_path.name)} "
        f"--patch-file /output/{sh_quote(patch_path.name)} "
        "--output /output/retrieval_results.json "
        "--content-output /output/retrieval_file_contents.json "
        f"{include_tests_arg}"
        "; test -s /output/retrieval_results.json; test -s /output/retrieval_file_contents.json"
    )
    logger.debug(f"Running {retrieval_source}/{retrieval_level} retrieval in container: {command}")
    result = container.exec_run(["bash", "-lc", command], demux=True)
    stdout, stderr = result.output
    stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
    stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
    if verbose_lsp_output:
        if stdout_text:
            logger.info(stdout_text)
        if stderr_text:
            logger.warning(stderr_text)
    if result.exit_code != 0:
        if not verbose_lsp_output:
            logger.warning(
                f"{retrieval_source}/{retrieval_level} retrieval output tail:\n"
                f"{compact_text(stdout_text + stderr_text)}"
            )
        raise RuntimeError(f"{retrieval_source}/{retrieval_level} retrieval failed with exit code {result.exit_code}")
    retrieval_results = json.loads((output_dir / "retrieval_results.json").read_text(encoding="utf-8"))
    return retrieval_results.get("selected_files", [])


def write_precomputed_retrieval_results(output_dir, retrieval_source, retrieval_level, retrieval_k, target_files):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schemaVersion": "retrieval-targets/v1",
        "mode": retrieval_source,
        "level": retrieval_level,
        "k": retrieval_k,
        "selected_files": target_files,
        "units": [
            {
                "rank": index,
                "id": file_path,
                "file": file_path,
                "name": file_path,
                "kind": "file",
                "score": None,
            }
            for index, file_path in enumerate(target_files)
        ],
    }
    (output_dir / "retrieval_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_headless_lsp_command(
    target_files,
    runner,
    output_name,
    summary_name,
    lsp_port,
    lsp_max_tokens,
    lsp_max_observations,
    lsp_max_snippet_lines,
    lsp_request_timeout_ms,
    lsp_total_timeout_ms,
    lsp_open_delay_ms,
):
    if isinstance(target_files, str):
        target_files = [target_files]
    if runner == "headless_basedpyright":
        server_expr = "/headless_lsp/node_modules/basedpyright/langserver.index.js"
    elif runner == "headless_pylance":
        server_expr = "$(find /root/.vscode-server/extensions -path '*/ms-python.vscode-pylance-*/dist/server.bundle.js' | head -n 1)"
    else:
        raise ValueError(f"Unsupported headless runner: {runner}")
    return (
        "set -e; "
        "NODE=$(find /root/.vscode-server/bin -maxdepth 2 -type f -name node | head -n 1); "
        f"SERVER={server_expr}; "
        "test -n \"$NODE\"; test -n \"$SERVER\"; "
        "\"$NODE\" /headless_lsp/extract_pylance_callgraph.js "
        "--workspace /testbed "
        f"--output /output/{sh_quote(output_name)} "
        f"--summary-output /output/{sh_quote(summary_name)} "
        "--python /opt/miniconda3/envs/testbed/bin/python "
        "--server \"$SERVER\" "
        "--transport socket "
        f"--lsp-port {int(lsp_port)} "
        f"--request-timeout-ms {int(lsp_request_timeout_ms)} "
        f"--total-timeout-ms {int(lsp_total_timeout_ms)} "
        f"--open-delay-ms {int(lsp_open_delay_ms)} "
        f"--max-tokens {int(lsp_max_tokens)} "
        "--token-fallback if-no-symbols "
        f"--max-observations {int(lsp_max_observations)} "
        f"--max-snippet-lines {int(lsp_max_snippet_lines)} "
        + " ".join(f"--file {sh_quote(target_file)}" for target_file in target_files)
        + " "
        f"; test -s /output/{sh_quote(output_name)}; test -s /output/{sh_quote(summary_name)}"
    )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def part_names(index):
    return (
        f"callgraph_raw.part{index:03d}.jsonl",
        f"callgraph_run_summary.part{index:03d}.json",
    )


def batch_names(index):
    return (
        f"callgraph_raw.batch{index:03d}.jsonl",
        f"callgraph_run_summary.batch{index:03d}.json",
    )


def empty_summary_totals(failed_files=0):
    return {
        "files": 0,
        "exceededFiles": 0,
        "incompleteFiles": 0,
        "budgetExceededFiles": 0,
        "tokenOccurrences": 0,
        "scannedTokens": 0,
        "observations": 0,
        "symbolTargets": 0,
        "snippetsRequested": 0,
        "snippetsResolved": 0,
        "snippetsTruncated": 0,
        "snippetFilesNotFound": 0,
        "warnings": {},
        "failedFiles": failed_files,
    }


def totals_for_file_entries(file_entries, failed_files=0):
    totals = empty_summary_totals(failed_files=failed_files)
    for entry in file_entries:
        report = entry.get("limitReport") or {}
        totals["files"] += 1
        if report.get("exceeded"):
            totals["exceededFiles"] += 1
        if report.get("incomplete"):
            totals["incompleteFiles"] += 1
        if report.get("budgetExceeded"):
            totals["budgetExceededFiles"] += 1
        token_scan = report.get("tokenScan") or {}
        totals["tokenOccurrences"] += int(token_scan.get("totalOccurrences", 0))
        totals["scannedTokens"] += int(token_scan.get("scanned", 0))
        observations = report.get("observations") or {}
        totals["observations"] += int(observations.get("produced", 0))
        symbol_targets = report.get("symbolTargets") or {}
        totals["symbolTargets"] += int(symbol_targets.get("total", 0))
        snippets = report.get("snippets") or {}
        totals["snippetsRequested"] += int(snippets.get("requested", 0))
        totals["snippetsResolved"] += int(snippets.get("resolved", 0))
        totals["snippetsTruncated"] += int(snippets.get("truncated", 0))
        totals["snippetFilesNotFound"] += int(snippets.get("fileNotFound", 0))
        for warning in report.get("warnings", []) or []:
            totals["warnings"][warning] = totals["warnings"].get(warning, 0) + 1
    return totals


def part_is_complete(output_dir, index, target_file):
    output_name, summary_name = part_names(index)
    part_raw = output_dir / output_name
    part_summary = output_dir / summary_name
    if not part_raw.exists() or not part_summary.exists():
        return False
    if part_raw.stat().st_size == 0 or part_summary.stat().st_size == 0:
        return False
    try:
        summary = load_json(part_summary)
    except Exception:
        return False
    files = summary.get("files", [])
    if len(files) != 1:
        return False
    relative_path = files[0].get("source", {}).get("relativePath")
    return relative_path == target_file


def materialize_batch_parts(output_dir, batch_items, batch_output_name, batch_summary_name):
    batch_raw = output_dir / batch_output_name
    batch_summary = output_dir / batch_summary_name
    if not batch_raw.exists() or not batch_summary.exists():
        return

    raw_by_file = {}
    for line in batch_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        rel_path = record.get("source", {}).get("relativePath")
        if rel_path:
            raw_by_file.setdefault(rel_path, []).append(line)

    summary = load_json(batch_summary)
    summary_by_file = {}
    for file_entry in summary.get("files", []) or []:
        rel_path = file_entry.get("source", {}).get("relativePath")
        if rel_path:
            summary_by_file.setdefault(rel_path, []).append(file_entry)

    for index, target_file in batch_items:
        output_name, summary_name = part_names(index)
        part_raw = output_dir / output_name
        part_summary = output_dir / summary_name
        raw_lines = raw_by_file.get(target_file, [])
        file_entries = summary_by_file.get(target_file, [])
        if raw_lines:
            part_raw.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        if file_entries:
            part_summary.write_text(
                json.dumps(
                    {
                        "schemaVersion": "headless-lsp-run-summary/v1",
                        "materializedFromBatch": batch_output_name,
                        "requestedFiles": [{"rank": index, "file": target_file}],
                        "failedFiles": [],
                        "totals": totals_for_file_entries(file_entries),
                        "files": file_entries,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )


def read_retrieval_results(output_dir):
    path = output_dir / "retrieval_results.json"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def write_retrieval_results(output_dir, result):
    (output_dir / "retrieval_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_retrieval_unit_ranks(output_dir, result):
    changed = False
    units = result.get("units", [])
    for index, unit in enumerate(units):
        if "rank" not in unit:
            unit["rank"] = index
            changed = True
    if changed:
        try:
            write_retrieval_results(output_dir, result)
        except OSError as exc:
            logger = logging.getLogger("swebench.harness.run_call_hier")
            logger.warning(f"Could not update retrieval_results.json with unit ranks: {exc}")
    return result


def retrieval_covers(output_dir, retrieval_source, retrieval_level, retrieval_k):
    result = read_retrieval_results(output_dir)
    if not result:
        return None
    if result.get("mode") != retrieval_source or result.get("level") != retrieval_level:
        return None
    if int(result.get("k", -1)) < int(retrieval_k):
        return None
    selected_files = result.get("selected_files", [])
    if retrieval_source != "oracle" and len(selected_files) < int(retrieval_k):
        return None
    if not selected_files:
        return None
    return ensure_retrieval_unit_ranks(output_dir, result)


def target_files_from_retrieval(retrieval_results, retrieval_k, file_limit):
    selected_files = retrieval_results.get("selected_files", [])[: int(retrieval_k)]
    if file_limit is not None:
        selected_files = selected_files[:file_limit]
    return selected_files


def all_parts_complete(output_dir, target_files):
    return bool(target_files) and all(
        part_is_complete(output_dir, index, target_file)
        for index, target_file in enumerate(target_files)
    )


def combine_headless_outputs(output_dir, target_files, part_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "callgraph_raw.jsonl"
    summary_output = output_dir / "callgraph_run_summary.json"
    records = []
    summaries = []
    failed_files = []
    for result in part_results:
        if result["exit_code"] != 0:
            failed_files.append(
                {
                    "rank": result["rank"],
                    "file": result["file"],
                    "exit_code": result["exit_code"],
                    "output": result["output_name"],
                    "summary": result["summary_name"],
                }
            )
            continue
        part_raw = output_dir / result["output_name"]
        part_summary = output_dir / result["summary_name"]
        if part_raw.exists():
            records.extend(line for line in part_raw.read_text(encoding="utf-8").splitlines() if line.strip())
        if part_summary.exists():
            summaries.append(json.loads(part_summary.read_text(encoding="utf-8")))
    raw_output.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")

    totals = {
        "files": 0,
        "exceededFiles": 0,
        "incompleteFiles": 0,
        "budgetExceededFiles": 0,
        "tokenOccurrences": 0,
        "scannedTokens": 0,
        "observations": 0,
        "symbolTargets": 0,
        "snippetsRequested": 0,
        "snippetsResolved": 0,
        "snippetsTruncated": 0,
        "snippetFilesNotFound": 0,
        "warnings": {},
        "failedFiles": len(failed_files),
    }
    files = []
    for summary in summaries:
        part_totals = summary.get("totals", {})
        for key in [
            "files",
            "exceededFiles",
            "incompleteFiles",
            "budgetExceededFiles",
            "tokenOccurrences",
            "scannedTokens",
            "observations",
            "symbolTargets",
            "snippetsRequested",
            "snippetsResolved",
            "snippetsTruncated",
            "snippetFilesNotFound",
        ]:
            totals[key] += int(part_totals.get(key, 0))
        for warning, count in part_totals.get("warnings", {}).items():
            totals["warnings"][warning] = totals["warnings"].get(warning, 0) + count
        files.extend(summary.get("files", []))
    combined_summary = {
        "schemaVersion": "headless-lsp-run-summary/v1",
        "combinedFromParts": True,
        "requestedFiles": [
            {"rank": index, "file": file_path}
            for index, file_path in enumerate(target_files)
        ],
        "failedFiles": failed_files,
        "totals": totals,
        "files": files,
    }
    summary_output.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schemaVersion": "callgraph-part-manifest/v1",
        "parts": [
            {
                "rank": result["rank"],
                "file": result["file"],
                "exitCode": result["exit_code"],
                "output": result["output_name"],
                "summary": result["summary_name"],
                "skipped": result.get("skipped", False),
                "complete": result["exit_code"] == 0 and part_is_complete(output_dir, result["rank"], result["file"]),
            }
            for result in part_results
        ],
    }
    (output_dir / "callgraph_part_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(records), failed_files


def combine_completed_parts(output_dir, target_files, logger):
    part_results = []
    for index, target_file in enumerate(target_files):
        output_name, summary_name = part_names(index)
        part_results.append(
            {
                "rank": index,
                "file": target_file,
                "exit_code": 0,
                "output_name": output_name,
                "summary_name": summary_name,
                "skipped": True,
            }
        )
    record_count, failed_files = combine_headless_outputs(output_dir, target_files, part_results)
    logger.debug(f"Recombined {record_count} existing callgraph records; failed files: {len(failed_files)}")


def prepare_initial_targets(
    instance,
    output_dir,
    retrieval_source,
    retrieval_level,
    effective_k,
    file_limit,
    logger,
):
    if retrieval_source == "precomputed_bm25":
        existing_retrieval = retrieval_covers(output_dir, retrieval_source, "file", effective_k)
        if existing_retrieval:
            logger.debug("Reusing existing retrieval_results.json")
            return target_files_from_retrieval(existing_retrieval, effective_k, file_limit)

        retrieved_files = parse_precomputed_bm25_files(instance)
        if not retrieved_files:
            logger.warning(f"Precomputed BM25 results not found in instance {instance['instance_id']}")
            return []
        retrieved_files = retrieved_files[:effective_k]
        write_precomputed_retrieval_results(
            output_dir=output_dir,
            retrieval_source=retrieval_source,
            retrieval_level="file",
            retrieval_k=effective_k,
            target_files=retrieved_files,
        )
        return retrieved_files[:file_limit] if file_limit is not None else retrieved_files

    if retrieval_source == "oracle" and retrieval_level == "file":
        existing_retrieval = retrieval_covers(output_dir, retrieval_source, retrieval_level, effective_k)
        if existing_retrieval:
            logger.debug("Reusing existing retrieval_results.json")
            return target_files_from_retrieval(existing_retrieval, effective_k, file_limit)

        retrieved_files = oracle_files_from_patch(instance)[:effective_k]
        write_precomputed_retrieval_results(
            output_dir=output_dir,
            retrieval_source=retrieval_source,
            retrieval_level=retrieval_level,
            retrieval_k=effective_k,
            target_files=retrieved_files,
        )
        return retrieved_files[:file_limit] if file_limit is not None else retrieved_files

    existing_retrieval = retrieval_covers(output_dir, retrieval_source, retrieval_level, effective_k)
    if existing_retrieval:
        logger.debug("Reusing existing container retrieval_results.json")
        return target_files_from_retrieval(existing_retrieval, effective_k, file_limit)
    return []


def run_headless_lsp(
    container,
    target_files,
    runner,
    output_dir,
    logger,
    lsp_max_tokens,
    lsp_max_observations,
    lsp_max_snippet_lines,
    lsp_file_workers,
    lsp_batch_size,
    lsp_request_timeout_ms,
    lsp_total_timeout_ms,
    lsp_open_delay_ms,
    verbose_lsp_output=False,
):
    def run_batch(batch_index, batch_items):
        if len(batch_items) == 1:
            output_name, summary_name = part_names(batch_items[0][0])
        else:
            output_name, summary_name = batch_names(batch_index)
        target_batch_files = [target_file for _, target_file in batch_items]
        command = make_headless_lsp_command(
            target_files=target_batch_files,
            runner=runner,
            output_name=output_name,
            summary_name=summary_name,
            lsp_port=2087 + batch_index,
            lsp_max_tokens=lsp_max_tokens,
            lsp_max_observations=lsp_max_observations,
            lsp_max_snippet_lines=lsp_max_snippet_lines,
            lsp_request_timeout_ms=lsp_request_timeout_ms,
            lsp_total_timeout_ms=lsp_total_timeout_ms,
            lsp_open_delay_ms=lsp_open_delay_ms,
        )
        logger.debug(
            f"Running {runner} batch={batch_index} files={len(batch_items)} in container: {command}"
        )
        result = container.exec_run(["bash", "-lc", command], demux=True)
        stdout, stderr = result.output
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        if verbose_lsp_output:
            if stdout_text:
                logger.info(stdout_text)
            if stderr_text:
                logger.warning(stderr_text)
        if result.exit_code == 0 and len(batch_items) > 1:
            materialize_batch_parts(output_dir, batch_items, output_name, summary_name)
        if result.exit_code != 0:
            logger.warning(
                f"{runner} extraction failed for batch={batch_index} files={target_batch_files} "
                f"with exit code {result.exit_code}; continuing"
            )
            if not verbose_lsp_output:
                logger.warning(
                    f"{runner} batch={batch_index} output tail:\n"
                    f"{compact_text(stdout_text + stderr_text)}"
                )
            if len(batch_items) > 1:
                logger.warning(
                    f"Retrying failed batch={batch_index} as {len(batch_items)} single-file LSP extractions"
                )
                retried_results = []
                for offset, item in enumerate(batch_items):
                    retried_results.extend(run_batch(batch_index * 1000 + offset + 1, [item]))
                return retried_results
        part_results = []
        for index, target_file in batch_items:
            part_output_name, part_summary_name = part_names(index)
            part_complete = part_is_complete(output_dir, index, target_file)
            part_results.append(
                {
                    "rank": index,
                    "file": target_file,
                    "exit_code": 0 if part_complete else (result.exit_code or 1),
                    "output_name": part_output_name,
                    "summary_name": part_summary_name,
                    "skipped": False,
                    "batch": batch_index,
                }
            )
        return part_results

    part_results = []
    pending_items = []
    for index, target_file in enumerate(target_files):
        output_name, summary_name = part_names(index)
        if part_is_complete(output_dir, index, target_file):
            logger.debug(f"Skipping completed LSP part rank={index} file={target_file}")
            part_results.append(
                {
                    "rank": index,
                    "file": target_file,
                    "exit_code": 0,
                    "output_name": output_name,
                    "summary_name": summary_name,
                    "skipped": True,
                }
            )
        else:
            pending_items.append((index, target_file))

    batch_size = max(1, int(lsp_batch_size))
    batches = [
        pending_items[start : start + batch_size]
        for start in range(0, len(pending_items), batch_size)
    ]
    worker_count = max(1, min(int(lsp_file_workers), len(batches) or 1))
    if worker_count == 1:
        for batch_index, batch_items in enumerate(batches):
            part_results.extend(run_batch(batch_index, batch_items))
    else:
        logger.debug(
            f"Running LSP extraction with {worker_count} parallel batch workers "
            f"and batch_size={batch_size}"
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_batch, batch_index, batch_items): (batch_index, batch_items)
                for batch_index, batch_items in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_index, batch_items = futures[future]
                try:
                    part_results.extend(future.result())
                except Exception as exc:
                    logger.exception(f"{runner} extraction crashed for batch={batch_index}: {exc}")
                    for index, target_file in batch_items:
                        output_name, summary_name = part_names(index)
                        part_results.append(
                            {
                                "rank": index,
                                "file": target_file,
                                "exit_code": 1,
                                "output_name": output_name,
                                "summary_name": summary_name,
                                "skipped": False,
                                "batch": batch_index,
                            }
                        )
    part_results.sort(key=lambda result: result["rank"])
    record_count, failed_files = combine_headless_outputs(output_dir, target_files, part_results)
    if record_count == 0:
        raise RuntimeError(f"{runner} extraction produced no records; failed files: {failed_files}")
    logger.debug(f"Combined {record_count} callgraph records; failed files: {len(failed_files)}")


def run_vscode_client(container, container_hash, bm_results, vscode_dir, logger):
    command = [
        f'"{vscode_dir}"',
        "--new-window --wait",
        '--profile "SWE-bench"',
        f'--folder-uri "vscode-remote://attached-container+{container_hash}/testbed"',
        *[
            f'--file-uri "vscode-remote://attached-container+{container_hash}/testbed/{bm_result}"'
            for bm_result in bm_results
        ],
    ]
    command = " ".join(command)
    logger.info(f"Running command: {command}")
    subprocess.run(command, shell=True)


def main(
    dataset_path,
    split,
    run_id,
    vscode_dir,
    mount_root,
    runner="headless_basedpyright",
    max_instances=None,
    instance_ids=None,
    instance_ids_file=None,
    file_limit=None,
    retrieval_source="precomputed_bm25",
    retrieval_level="file",
    retrieval_k=None,
    include_tests=False,
    dry_run=False,
    skip_env_build=False,
    namespace=None,
    instance_image_tag="latest",
    env_image_tag="latest",
    force_reextract=False,
    lsp_max_tokens=2500,
    lsp_max_observations=800,
    lsp_max_snippet_lines=1600,
    lsp_file_workers=1,
    lsp_batch_size=10,
    lsp_request_timeout_ms=15000,
    lsp_total_timeout_ms=1200000,
    lsp_open_delay_ms=1000,
    instance_workers=1,
    verbose_lsp_output=False,
):
    logger = logging.getLogger("swebench.harness.run_call_hier")
    for noisy_logger in [
        "httpx",
        "datasets",
        "datasets.load",
        "datasets.packaged_modules.cache.cache",
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    instance_ids = set(instance_ids or [])
    if instance_ids_file:
        with open(instance_ids_file, encoding="utf-8") as f:
            instance_ids.update(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    dataset = load_swebench_dataset(dataset_path, split, instance_ids=instance_ids or None)
    if max_instances is not None:
        dataset = dataset[:max_instances]

    client = docker.from_env()
    logger.debug(f"Docker host: {client.api.base_url}")

    build_base_images(
        client,
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )

    if not skip_env_build:
        build_env_images(
            client,
            dataset,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )

    instances_to_process = [
        instance
        for instance in dataset
        if should_include(instance["instance_id"], instance_ids)
    ]
    if max_instances is not None:
        instances_to_process = instances_to_process[:max_instances]

    def process_instance(instance):
        instance_id = instance["instance_id"]
        logger.debug(f"Processing instance {instance_id}")

        effective_k = retrieval_k or file_limit or 20
        output_dir = Path(mount_root) / "output" / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if force_reextract:
            for pattern in [
                "callgraph_raw*.jsonl",
                "callgraph_run_summary*.json",
                "callgraph_part_manifest.json",
                "retrieval_results.json",
                "retrieval_file_contents.json",
            ]:
                for path in output_dir.glob(pattern):
                    path.unlink()
        try:
            target_files = prepare_initial_targets(
                instance=instance,
                output_dir=output_dir,
                retrieval_source=retrieval_source,
                retrieval_level=retrieval_level,
                effective_k=effective_k,
                file_limit=file_limit,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Error preparing retrieval targets for instance {instance_id}: {e}")
            return "failed", "prepare_targets"
        if not target_files and retrieval_source in {"precomputed_bm25", "oracle"} and retrieval_level == "file":
            logger.warning(f"No retrieval target files found for instance {instance_id}")
            return "failed", "no_targets"
        logger.debug(
            f"Retrieval source={retrieval_source} level={retrieval_level} k={effective_k}"
        )
        logger.debug(f"Initial target files: {target_files}")
        if dry_run:
            return "ok", "dry_run"
        if runner in {"headless_basedpyright", "headless_pylance"} and all_parts_complete(output_dir, target_files):
            logger.debug("All requested LSP parts already exist; recombining without starting an instance container")
            combine_completed_parts(output_dir, target_files, logger)
            return "skipped", "already_complete"

        volumes = {
            f"{mount_root}/.vscode-server": {"bind": "/root/.vscode-server", "mode": "rw"},
            f"{mount_root}/.vscode": {"bind": "/testbed/.vscode", "mode": "rw"},
            f"{mount_root}/output/{instance_id}": {"bind": "/output", "mode": "rw"},
            f"{mount_root}/headless_lsp": {"bind": "/headless_lsp", "mode": "ro"},
        }

        test_spec = make_test_spec(
            instance,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )

        container = None
        worker_client = docker.from_env()
        try:
            container = build_container(
                test_spec,
                worker_client,
                run_id,
                logger,
                nocache=False,
                volumes=volumes,
            )
            container.start()
            container_hash = ('/' + container.name).encode('utf-8').hex()
            if retrieval_source in {"bm25"} or (retrieval_source == "oracle" and retrieval_level == "function"):
                existing_retrieval = retrieval_covers(output_dir, retrieval_source, retrieval_level, effective_k)
                if existing_retrieval:
                    target_files = target_files_from_retrieval(existing_retrieval, effective_k, file_limit)
                    logger.debug("Reusing existing container retrieval_results.json")
                else:
                    target_files = run_container_retrieval(
                        container=container,
                        instance=instance,
                        output_dir=output_dir,
                        retrieval_source=retrieval_source,
                        retrieval_level=retrieval_level,
                        retrieval_k=effective_k,
                        include_tests=include_tests,
                        logger=logger,
                        verbose_lsp_output=verbose_lsp_output,
                    )
                    if file_limit is not None:
                        target_files = target_files[:file_limit]
                logger.debug(f"Container retrieval target files: {target_files}")
                if not target_files:
                    logger.warning(f"No container retrieval target files found for instance {instance_id}")
                    return "failed", "empty_retrieval"
            if runner in {"headless_basedpyright", "headless_pylance"}:
                run_headless_lsp(
                    container,
                    target_files,
                    runner,
                    output_dir,
                    logger,
                    lsp_max_tokens=lsp_max_tokens,
                    lsp_max_observations=lsp_max_observations,
                    lsp_max_snippet_lines=lsp_max_snippet_lines,
                    lsp_file_workers=lsp_file_workers,
                    lsp_batch_size=lsp_batch_size,
                    lsp_request_timeout_ms=lsp_request_timeout_ms,
                    lsp_total_timeout_ms=lsp_total_timeout_ms,
                    lsp_open_delay_ms=lsp_open_delay_ms,
                    verbose_lsp_output=verbose_lsp_output,
                )
            elif runner == "vscode":
                run_vscode_client(container, container_hash, target_files, vscode_dir, logger)
            else:
                raise ValueError(f"Unknown runner: {runner}")
            return "ok", "extracted"
        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {e}")
            return "failed", str(e).replace("\n", " ")[:120]
        finally:
            if container is not None:
                cleanup_container(worker_client, container, logger)
            # remove_image(client, test_spec.instance_image_key, logger)
            try:
                worker_client.close()
            except Exception:
                pass

    worker_count = max(1, min(int(instance_workers), len(instances_to_process) or 1))
    print(
        f"[call-hier] total={len(instances_to_process)} "
        f"instance_workers={worker_count} lsp_batch_size={lsp_batch_size} "
        f"lsp_file_workers={lsp_file_workers}",
        flush=True,
    )
    progress = ProgressTracker(len(instances_to_process))
    if worker_count == 1:
        for instance in instances_to_process:
            instance_id = instance["instance_id"]
            status, detail = process_instance(instance)
            progress.update(instance_id, status, detail)
    else:
        logger.debug(
            f"Running {len(instances_to_process)} instances with {worker_count} parallel instance workers"
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_instance, instance): instance["instance_id"]
                for instance in instances_to_process
            }
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    status, detail = future.result()
                except Exception as exc:
                    logger.exception(f"Unhandled error processing instance {instance_id}: {exc}")
                    status, detail = "failed", str(exc).replace("\n", " ")[:120]
                progress.update(instance_id, status, detail)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract SWE-bench Oracle / BM25 retrieval results' function calls",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="princeton-nlp/SWE-bench_Lite_bm25_13K",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split of the dataset.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="test",
        help="Run ID for the container.",
    )
    parser.add_argument(
        "--vscode_dir",
        type=str,
        default="code",
        help="Path to the VS Code executable. Only used when --runner vscode.",
    )
    parser.add_argument(
        "--runner",
        choices=["headless_basedpyright", "headless_pylance", "vscode"],
        default="headless_basedpyright",
        help="Extraction runner. headless_basedpyright avoids opening a VS Code client UI.",
    )
    parser.add_argument(
        "--mount_root",
        type=str,
        default="/mnt/sweb",
        help="Mount root for the container.",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum number of matching instances to process.",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="*",
        default=None,
        help="Optional instance IDs to process.",
    )
    parser.add_argument(
        "--instance_ids_file",
        type=str,
        default=None,
        help="Optional newline-delimited instance IDs to process.",
    )
    parser.add_argument(
        "--file_limit",
        type=int,
        default=None,
        help="Maximum number of retrieved files to open per instance.",
    )
    parser.add_argument(
        "--retrieval_source",
        choices=["precomputed_bm25", "bm25", "oracle"],
        default="precomputed_bm25",
        help=(
            "Where target files come from. precomputed_bm25 parses existing "
            "SWE-bench *_bm25_* text fields; bm25 builds a lightweight BM25 "
            "ranking inside /testbed; oracle uses the gold patch."
        ),
    )
    parser.add_argument(
        "--retrieval_level",
        choices=["file", "function"],
        default="file",
        help=(
            "Retrieval granularity. function mode ranks AST class/function units "
            "and then opens the containing files for LSP call hierarchy extraction."
        ),
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=None,
        help="Number of files or function units to retrieve before optional --file_limit.",
    )
    parser.add_argument(
        "--include_tests",
        action="store_true",
        help="Include test files in container-side BM25/function retrieval.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only parse target files from the dataset; do not start containers or VS Code.",
    )
    parser.add_argument(
        "--skip_env_build",
        action="store_true",
        help="Skip env image build/check. Use when env images are already built.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help='Docker image namespace. Use "swebench" for Docker Hub images.',
    )
    parser.add_argument(
        "--instance_image_tag",
        type=str,
        default="latest",
        help="Instance image tag.",
    )
    parser.add_argument(
        "--env_image_tag",
        type=str,
        default="latest",
        help="Environment image tag.",
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Remove previous retrieval/callgraph outputs for matching instances before extraction.",
    )
    parser.add_argument(
        "--lsp_max_tokens",
        type=int,
        default=2500,
        help=(
            "Maximum token occurrences scanned by token fallback. This only "
            "matters when --token-fallback is active, currently if no document "
            "symbols are available."
        ),
    )
    parser.add_argument(
        "--lsp_max_observations",
        type=int,
        default=800,
        help="Maximum call hierarchy observations retained per retrieved file.",
    )
    parser.add_argument(
        "--lsp_max_snippet_lines",
        type=int,
        default=1600,
        help="Maximum source lines retained per resolved symbol/call snippet.",
    )
    parser.add_argument(
        "--lsp_file_workers",
        type=int,
        default=1,
        help=(
            "Number of LSP batches to extract in parallel inside one SWE-bench "
            "instance container. Use 1 to maximize LSP cache reuse; use 2-4 only "
            "when CPU and memory headroom are ample."
        ),
    )
    parser.add_argument(
        "--lsp_batch_size",
        type=int,
        default=10,
        help=(
            "Number of retrieved files handled by one LSP server process. Larger "
            "batches reduce repeated project parsing; smaller batches improve "
            "fault isolation."
        ),
    )
    parser.add_argument(
        "--lsp_request_timeout_ms",
        type=int,
        default=15000,
        help="Timeout for one LSP JSON-RPC request.",
    )
    parser.add_argument(
        "--lsp_total_timeout_ms",
        type=int,
        default=1200000,
        help="Wall-clock timeout for one headless LSP batch process.",
    )
    parser.add_argument(
        "--lsp_open_delay_ms",
        type=int,
        default=1000,
        help="Delay after opening each file before requesting document symbols.",
    )
    parser.add_argument(
        "--instance_workers",
        type=int,
        default=1,
        help=(
            "Number of SWE-bench instances to process in parallel. Each worker "
            "starts its own container; combine with --lsp_batch_size to reuse one "
            "LSP server per instance."
        ),
    )
    parser.add_argument(
        "--verbose_lsp_output",
        action="store_true",
        help=(
            "Print full container retrieval and headless LSP stdout/stderr. "
            "By default, successful batch output is hidden and failures show only a compact tail."
        ),
    )

    args = parser.parse_args()
    main(**vars(args))
