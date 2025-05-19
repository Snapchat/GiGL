import datetime
import getpass
import os
import pathlib
import subprocess
import tempfile

import yaml

from gigl.common import UriFactory
from gigl.src.common.utils.file_loader import FileLoader

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


def assert_bq_dataset_exists(dataset_name: str, project: str):
    command = f"bq show --project_id {project} {dataset_name}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        raise ValueError(
            f"BigQuery dataset '{dataset_name}' does not exist in project '{project}' or you do not have access to it."
        )
    print(f"BigQuery dataset '{dataset_name}' exists.")


def assert_gcs_bucket_exists(bucket_name: str):
    command = f"gsutil ls gs://{bucket_name}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
        raise ValueError(
            f"GCS bucket '{bucket_name}' does not exist or you do not have access to it."
        )
    print(f"GCS bucket '{bucket_name}' exists.")


def main():
    # Source file path

    print("Welcome to the GiGL Cloud Environment Configuration Script!")
    print("This script will help you set up your cloud environment for GiGL.")
    print(
        "Before running this script, please ensure you have followed the GiGL Cloud Setup Guide:"
    )
    print(
        "https://snapchat.github.io/GiGL/docs/user_guide/getting_started/cloud_setup_guide"
    )
    print("======================================================")

    print(
        f"We will use `{TEMPLATE_SOURCE_RESOURCE_CONFIG}` as the template for your resource config YAML file."
    )

    # Default values
    print("Please provide us Project information.")
    project: str = input("GCP Project name that you are planning on using: ").strip()
    assert_gcp_project_exists(project)
    region: str = input(
        f"The GCP region where you created your resources i.e. `us-central1`: "
    ).strip()
    assert region, "Region cannot be empty"
    gcp_service_account_email: str = input("The GCP Service account: ").strip()
    assert_service_account_exists(gcp_service_account_email, project)

    print("Please provide us the information on the BQ Datasets you want to use")
    temp_assets_bq_dataset_name: str = input(
        "`temp_assets_bq_dataset_name` - Dataset name used for temporary assets: "
    ).strip()
    assert_bq_dataset_exists(dataset_name=temp_assets_bq_dataset_name, project=project)
    embedding_bq_dataset_name: str = input(
        f"`embedding_bq_dataset_name` - Dataset used of output embeddings: "
    ).strip()
    assert_bq_dataset_exists(dataset_name=embedding_bq_dataset_name, project=project)

    print("Please provide us the information on the GCS Buckets you want to use")
    temp_assets_bucket: str = input(
        f"`temp_assets_bucket` - GCS Bucket for storing temporary assets: "
    ).strip()
    assert_gcs_bucket_exists(bucket_name=temp_assets_bucket)
    perm_assets_bucket: str = input(
        f"`perm_assets_bucket` - GCS Bucket for storing permanent assets: "
    ).strip()
    assert_gcs_bucket_exists(bucket_name=perm_assets_bucket)

    curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    curr_username = getpass.getuser()
    resource_config_dest_path = f"gs://{perm_assets_bucket}/{curr_username}/{curr_datetime}/gigl_test_default_resource_config.yaml"

    destination_file_path = (
        input(
            f"Output path for resource config (default: {resource_config_dest_path}); can be GCS or Local path: "
        ).strip()
        or resource_config_dest_path
    )

    print("=======================================================")
    print(f"Will now create the resource config file @ {destination_file_path}.")
    print("Using the following values:")
    update_fields_dict = {
        "project": project,
        "region": region,
        "gcp_service_account_email": gcp_service_account_email,
        "temp_assets_bucket": temp_assets_bucket,
        "temp_regional_assets_bucket": temp_assets_bucket,
        "perm_assets_bucket": perm_assets_bucket,
        "temp_assets_bq_dataset_name": temp_assets_bq_dataset_name,
        "embedding_bq_dataset_name": embedding_bq_dataset_name,
    }
    for key, value in update_fields_dict.items():
        print(f"  {key}: {value}")

    with open(TEMPLATE_SOURCE_RESOURCE_CONFIG, "r") as file:
        config = yaml.safe_load(file)

    # # Update the YAML content
    common_compute_config: dict = config.get("shared_resource_config").get(
        "common_compute_config"
    )
    common_compute_config.update(update_fields_dict)

    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp_file.name, "w") as file:
        yaml.safe_dump(config, file)

    file_loader = FileLoader(project=project)
    file_uri_src = UriFactory.create_uri(uri=tmp_file.name)
    file_uri_dest = UriFactory.create_uri(uri=destination_file_path)
    file_loader.load_file(file_uri_src=file_uri_src, file_uri_dst=file_uri_dest)

    print(f"Updated YAML file saved at '{destination_file_path}'")

    # Update the user's shell configuration
    should_update_shell_config = (
        input(
            "Do you want to update your shell configuration file so you can use this new resource config for tests? [y/n] (Default: y): "
        )
        .strip()
        .lower()
        or "y"
    )
    if should_update_shell_config == "y":
        shell_config_path: str = infer_shell_file()
        update_shell_config(shell_config_path, destination_file_path, project)
    else:
        print(
            "Skipping shell configuration update. Please remember to set the environment variables manually "
            + "if you want `make unit_test | integration_test` commands to work correctly."
        )


if __name__ == "__main__":
    main()
