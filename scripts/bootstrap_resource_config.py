import datetime
import getpass
import os
import pathlib
import subprocess

import yaml
from google.cloud import (  # Ensure required libraries are installed: pip install google-cloud-storage google-cloud-bigquery google-cloud-iam
    bigquery,
    storage,
)

from gigl.common.utils.gcs import GcsUtils
from gigl.src.common.utils.bq import BqUtils

GIGL_ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_SOURCE_RESOURCE_CONFIG = (
    GIGL_ROOT_DIR / "deployment" / "configs" / "unittest_resource_config.yaml"
)


def infer_shell_file() -> str:
    """Infers the user's default shell configuration file."""
    shell = os.environ.get("SHELL", "")
    shell_config_map = {
        "zsh": "~/.zshrc",
        "bash": "~/.bashrc",
    }

    for key, config_file in shell_config_map.items():
        if key in shell:
            print(f"Detected shell: {key}. Using config file: {config_file}")
            return config_file

    print(
        "Could not infer the default shell. Please specify the shell configuration file manually."
    )
    return input(
        "Enter the path to your shell configuration file (e.g., ~/.bashrc): "
    ).strip()


def update_shell_config(
    shell_config_path, gigl_test_default_resource_config, gigl_project
):
    """Updates the shell configuration file with the environment variables in an idempotent way."""
    shell_config_path = os.path.expanduser(shell_config_path)
    start_marker = "# ====== GiGL ENV Config - Begin ====="
    end_marker = "# ====== GiGL ENV Config - End====="
    export_lines = [
        start_marker,
        f"export GIGL_TEST_DEFAULT_RESOURCE_CONFIG={gigl_test_default_resource_config}",
        f"export GIGL_PROJECT={gigl_project}",
        end_marker,
    ]

    # Read the existing shell config file
    if not os.path.exists(shell_config_path):
        raise FileNotFoundError(
            f"Shell config file '{shell_config_path}' does not exist."
        )
    shell_config_lines: list[str]
    with open(shell_config_path, "r") as shell_config:
        shell_config_lines = shell_config.readlines()

    inside_block = False
    updated_shell_config_lines = []
    for line in shell_config_lines:
        if line.strip() == start_marker:
            inside_block = True
        elif line.strip() == end_marker:
            inside_block = False
            continue
        if not inside_block:
            updated_shell_config_lines.append(line)

    # Add the new GiGL config block
    updated_shell_config_lines.extend(export_lines)

    # Write back to the shell config file
    with open(shell_config_path, "w") as shell_config:
        shell_config.writelines(updated_shell_config_lines)

    print(f"Updated {shell_config_path} with: {updated_shell_config_lines}")


def assert_gcp_project_exists(project_id: str):
    command = f"gcloud projects describe {project_id}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        raise ValueError(
            f"Project '{project_id}' does not exist or you do not have access to it."
        )


def assert_service_account_exists(service_account_email: str, project: str):
    # Check if the service account exists
    command = f"gcloud iam service-accounts describe {service_account_email} --project={project}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        raise ValueError(
            f"Service account '{service_account_email}' does not exist or you do not have access to it."
        )
    print(f"Service account '{service_account_email}' exists.")


def main():
    # Source file path

    print("Welcome to the GiGL Cloud Environment Configuration Script!")
    print("This script will help you set up your cloud environment for GiGL.")
    print("======================================================")

    print(
        f"We will use `{TEMPLATE_SOURCE_RESOURCE_CONFIG}` as the template for your resource config YAML file."
    )
    with open(TEMPLATE_SOURCE_RESOURCE_CONFIG, "r") as file:
        config = yaml.safe_load(file)

    # Default values
    _DEFAULT_REGION = "us-central1"

    project = input("Enter the value for project: ").strip()
    assert_gcp_project_exists(project)
    region: str = (
        input(f"Enter the value for region (default: {_DEFAULT_REGION}): ").strip()
        or _DEFAULT_REGION
    )

    print(f"Project '{project}' exists.")

    bq_utils = BqUtils(project)
    gcs_utils = GcsUtils(project)

    # Ask user for input values
    temp_assets_gcs_bucket: str = input(
        f"Enter the value for temp_assets_gcs_bucket: "
    ).strip()
    assert gcs_utils.does_gcs_bucket_exist(
        temp_assets_gcs_bucket
    ), f"Bucket '{temp_assets_gcs_bucket}' does not exist or you do not have access to it."
    perm_assets_gcs_bucket: str = (
        input(
            f"Enter the value for perm_assets_gcs_bucket (default: {temp_assets_gcs_bucket}): "
            + f"Please "
        ).strip()
        or temp_assets_gcs_bucket
    )
    assert gcs_utils.does_gcs_bucket_exist(
        perm_assets_gcs_bucket
    ), f"Bucket '{perm_assets_gcs_bucket}' does not exist or you do not have access to it."

    temp_assets_bq_dataset_name: str = input(
        "Enter the value for temp_assets_bq_dataset_name: "
    ).strip()
    assert bq_utils.does_bq_table_exist(
        temp_assets_bq_dataset_name
    ), f"Dataset '{temp_assets_bq_dataset_name}' does not exist or you do not have access to it."
    embedding_bq_dataset_name: str = (
        input(
            f"Enter the value for embedding_bq_dataset_name (default: {temp_assets_bq_dataset_name}): "
        ).strip()
        or temp_assets_bq_dataset_name
    )
    assert bq_utils.does_bq_table_exist(
        embedding_bq_dataset_name
    ), f"Dataset '{embedding_bq_dataset_name}' does not exist or you do not have access to it."

    gcp_service_account_email: str = input(
        "Enter the value for gcp_service_account_email: "
    ).strip()
    assert_service_account_exists(gcp_service_account_email, project)

    curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    curr_username = getpass.getuser()
    default_destination_file_uri = f"gs://{perm_assets_gcs_bucket}/{curr_username}/{curr_datetime}/gigl_test_default_resource_config.yaml"

    destination_file_uri = (
        input(
            f"Enter the destination file URI (default: {default_destination_file_uri}): "
        ).strip()
        or default_destination_file_uri
    )

    # # Update the YAML content
    common_compute_config = config.get("shared_resource_config", {}).get(
        "common_compute_config", {}
    )
    common_compute_config["project"] = project
    common_compute_config["region"] = region
    common_compute_config["gcp_service_account_email"] = gcp_service_account_email
    common_compute_config["temp_assets_bucket"] = temp_assets_gcs_bucket
    common_compute_config["temp_regional_assets_bucket"] = temp_assets_gcs_bucket
    common_compute_config["perm_assets_bucket"] = perm_assets_gcs_bucket
    common_compute_config["temp_assets_bq_dataset_name"] = temp_assets_bq_dataset_name
    common_compute_config["embedding_bq_dataset_name"] = embedding_bq_dataset_name

    # # Save the updated YAML file
    # with open(destination_file, "w") as file:
    #     yaml.safe_dump(config, file)

    # print(f"Updated YAML file saved at '{destination_file}'")

    # # Update the user's shell configuration
    # shell_config_path = infer_default_shell()
    # update_shell_config(shell_config_path, destination_file, project)


if __name__ == "__main__":
    main()
