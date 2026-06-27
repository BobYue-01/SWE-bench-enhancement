from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess
import logging
import docker
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


def sh_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def should_include(instance_id, instance_ids):
    return not instance_ids or instance_id in instance_ids


def run_headless_lsp(container, bm_results, runner, logger):
    files_args = " ".join(f"--file {sh_quote(path)}" for path in bm_results)
    if runner == "headless_basedpyright":
        server_expr = "/headless_lsp/node_modules/basedpyright/langserver.index.js"
    elif runner == "headless_pylance":
        server_expr = "$(find /root/.vscode-server/extensions -path '*/ms-python.vscode-pylance-*/dist/server.bundle.js' | head -n 1)"
    else:
        raise ValueError(f"Unsupported headless runner: {runner}")
    command = (
        "set -e; "
        "NODE=$(find /root/.vscode-server/bin -maxdepth 2 -type f -name node | head -n 1); "
        f"SERVER={server_expr}; "
        "test -n \"$NODE\"; test -n \"$SERVER\"; "
        "\"$NODE\" /headless_lsp/extract_pylance_callgraph.js "
        "--workspace /testbed "
        "--output /output/callgraph_raw.jsonl "
        "--python /opt/miniconda3/envs/testbed/bin/python "
        "--server \"$SERVER\" "
        "--transport socket "
        "--lsp-port 2087 "
        "--request-timeout-ms 5000 "
        "--total-timeout-ms 120000 "
        "--open-delay-ms 1000 "
        "--max-tokens 250 "
        "--token-fallback if-no-symbols "
        "--max-observations 40 "
        "--max-snippet-lines 80 "
        f"{files_args}"
        "; test -s /output/callgraph_raw.jsonl; test -s /output/callgraph_run_summary.json"
    )
    logger.info(f"Running {runner} command in container: {command}")
    result = container.exec_run(["bash", "-lc", command], demux=True)
    stdout, stderr = result.output
    if stdout:
        logger.info(stdout.decode("utf-8", errors="replace"))
    if stderr:
        logger.error(stderr.decode("utf-8", errors="replace"))
    if result.exit_code != 0:
        raise RuntimeError(f"{runner} extraction failed with exit code {result.exit_code}")


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
    file_limit=None,
    dry_run=False,
    skip_env_build=False,
):
    logger = logging.getLogger("swebench.harness.run_call_hier")
    instance_ids = set(instance_ids or [])
    dataset = load_swebench_dataset(dataset_path, split, instance_ids=instance_ids or None)
    if max_instances is not None:
        dataset = dataset[:max_instances]

    client = docker.from_env()
    logger.info(f"Docker host: {client.api.base_url}")

    build_base_images(
        client,
        dataset,
        instance_image_tag = "latest",
        env_image_tag = "latest",
    )

    if not skip_env_build:
        build_env_images(
            client,
            dataset,
            instance_image_tag = "latest",
            env_image_tag = "latest",
        )

    processed_count = 0
    for instance in dataset:
        instance_id = instance["instance_id"]
        if not should_include(instance_id, instance_ids):
            continue
        if max_instances is not None and processed_count >= max_instances:
            break
        logger.info(f"Processing instance {instance_id}")

        code_blocks = re.findall(r"\[end of (?i:readme)(?:[^\]\n]*)?\]\n(.*?)\n<\/code>", instance["text"], re.DOTALL)
        if not code_blocks:
            logger.error(f"Code blocks not found in instance {instance_id}")
            continue
        bm_results = re.findall(r"\n\[start of (.*?)\]\n", code_blocks[0])
        if not bm_results:
            logger.warning(f"BM results not found in instance {instance_id}")
            continue
        if file_limit is not None:
            bm_results = bm_results[:file_limit]
        logger.info(f"BM results: {bm_results}")
        if dry_run:
            processed_count += 1
            continue

        volumes = {
            f"{mount_root}/.vscode-server": {"bind": "/root/.vscode-server", "mode": "rw"},
            f"{mount_root}/.vscode": {"bind": "/testbed/.vscode", "mode": "rw"},
            f"{mount_root}/output/{instance_id}": {"bind": "/output", "mode": "rw"},
            f"{mount_root}/headless_lsp": {"bind": "/headless_lsp", "mode": "ro"},
        }

        test_spec = make_test_spec(instance)
        container = build_container(
            test_spec,
            client,
            run_id,
            logger,
            nocache=False,
            volumes=volumes,
        )

        try:
            container.start()
            container_hash = ('/' + container.name).encode('utf-8').hex()
            if runner in {"headless_basedpyright", "headless_pylance"}:
                run_headless_lsp(container, bm_results, runner, logger)
            elif runner == "vscode":
                run_vscode_client(container, container_hash, bm_results, vscode_dir, logger)
            else:
                raise ValueError(f"Unknown runner: {runner}")
        finally:
            cleanup_container(client, container, logger)
            # remove_image(client, test_spec.instance_image_key, logger)
        processed_count += 1


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
        "--file_limit",
        type=int,
        default=None,
        help="Maximum number of retrieved files to open per instance.",
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

    args = parser.parse_args()
    main(**vars(args))
