from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess
import logging
import docker
from swebench.harness.docker_build import (
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


def main(dataset_path, split, run_id, vscode_dir):
    logger = logging.getLogger("swebench.harness.run_call_hier")
    dataset = load_swebench_dataset(dataset_path, split)

    client = docker.from_env()

    build_env_images(client, dataset, force_rebuild=False, max_workers=1)

    for instance in dataset:
        test_spec = make_test_spec(
            instance, namespace="swebench", instance_image_tag="latest"
        )

        instance_id = instance["instance_id"]
        logger.info(f"Processing instance {instance_id}")

        code_blocks = re.findall(r"\[end of (?i:readme)(?:[^\]\n]*)?\]\n(.*?)\n<\/code>", instance["text"], re.DOTALL)
        if not code_blocks:
            logger.error(f"Code blocks not found in instance {instance_id}")
            continue
        bm_results = re.findall(r"\n\[start of (.*?)\]\n", code_blocks[0])
        if not bm_results:
            logger.warning(f"BM results not found in instance {instance_id}")
            continue
        logger.info(f"BM results: {bm_results}")

        volumes = {
            "/mnt/sweb/.vscode-server": {"bind": "/root/.vscode-server", "mode": "rw"},
            "/mnt/sweb/.vscode": {"bind": "/testbed/.vscode", "mode": "rw"},
            f"/mnt/sweb/output/{instance_id}": {"bind": "/output", "mode": "rw"},
        }

        container = build_container(
            test_spec,
            client,
            run_id,
            logger,
            nocache=False,
            volumes=volumes,
        )

        container.start()

        container_hash = container.name.encode('utf-8').hex()

        # unknown problem when breaking the shell command into a sequence of arguments
        # therefore, we need to use a joined string to make it work
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

        cleanup_container(client, container, logger)
        remove_image(client, test_spec.instance_image_key, logger)


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
        required=True,
        help="Path to the VS Code directory. "
        "E.g. /mnt/c/Users/<username>/AppData/Local/Programs/Microsoft VS Code/bin/code",
    )

    args = parser.parse_args()
    main(**vars(args))
