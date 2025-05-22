import argparse
import getpass
import inspect
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Param:
    default: Optional[str]
    description: str
    long_description: str = ""
    required: bool = True


class SupportedParams:
    defaults: Dict[str, Param]

    def __init__(self):
        whoami = getpass.getuser()
        try:
            project = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"], text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            print(
                "Error retrieving active project name; is your gcloud SDK configured correctly?",
                e,
            )
            raise
        self.defaults: Dict[str, Param] = {
            "project": Param(default=project, description="GCP project ID"),
            "zone": Param(default="us-central1-a", description="GCP zone"),
            "service_account": Param(default=None, description="Service account email"),
            "machine_name": Param(
                default=f"{whoami}-gigl-dev-instance",
                description="Name of the VM instance",
            ),
            "machine_type": Param(default="n1-highmem-32", description="Machine type"),
            "accelerator_type": Param(
                default="nvidia-tesla-t4", description="GPU accelerator type"
            ),
            "accelerator_count": Param(
                default="4", description="Number of GPUs to attach to the VM"
            ),
            "machine_boot_image": Param(
                default="projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda",
                description="Boot image for the VM.",
                long_description="You can find images google provides @ https://cloud.google.com/deep-learning-vm/docs/images. \n"
                + "Please ensure to use an image whose cuda version is compatible with GiGL.\n"
                + "See the following for GiGL supported cuda versions: "
                + "https://snapchat.github.io/GiGL/docs/user_guide/getting_started/installation.html#supported-environments",
            ),
            "boot_drive_size_gb": Param(
                default="1000", description="Boot disk size in GB"
            ),
            "labels": Param(
                default=None,
                description="Labels for the VM instance in the form of key=value,key2=value2",
                required=False,
            ),
        }


class GCPInstanceOpsAgentPolicyCreator:
    """
    Class to create an OPS agent policy for GCP instances.

    The Ops Agent is the primary agent for collecting telemetry from your Compute Engine instances.
    Combining the collection of logs, metrics, and traces into a single process

    Agent policies enable automated installation and maintenance of the Ops Agent across a fleet of
    Compute Engine VMs that match user-specified criteria. In this case, the user-specified criteria
    is the inclusion of a specific label on the VM instance defined by
    the label key: :attr:`GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_KEY`
    and label value: :attr:`GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_VALUE`.

    Any time an instance is created with this label, the Ops Agent will be installed and configured.

    For more details, see:
    https://cloud.google.com/stackdriver/docs/solutions/agents/ops-agent/agent-policies-overview
    """

    POLICY_INCLUSION_LABEL_KEY: str = "goog-ops-agent-policy"
    POLICY_INCLUSION_LABEL_VALUE: str = "v2-x86-template-1-4-0"

    POLICY: str = inspect.cleandoc(
        f"""
    agentsRule:
        packageState: installed
        version: latest
    instanceFilter:
        inclusionLabels:
        - labels:
            {POLICY_INCLUSION_LABEL_KEY}: {POLICY_INCLUSION_LABEL_VALUE}
    """
    )

    @staticmethod
    def get_policy_ops_agent_policy_name(zone: str) -> str:
        """
        Get the OPS agent policy name for the given zone.
        """
        return f"goog-ops-agent-v2-x86-template-1-4-0-{zone}"

    @staticmethod
    def get_existing_ops_agent_policy(project: str, zone: str) -> Optional[str]:
        """
        Get the existing OPS agent policy for the given project and zone.
        Checks for the default policy name 'goog-ops-agent-v2-x86-template-1-4-0-us-central1-a'.
        Returns the policy name if it exists, else None.
        """
        policy_name = GCPInstanceOpsAgentPolicyCreator.get_policy_ops_agent_policy_name(
            zone
        )
        found_policy = False
        cmd = (
            f"gcloud compute instances ops-agents policies describe {policy_name} "
            + f"--project={project} --zone={zone}"
        )
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            if (
                f"{GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_KEY}: {GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_VALUE}"
                in output
            ):
                found_policy = True
        except subprocess.CalledProcessError as e:
            print("OPS agent policy not found or error retrieving:", e)

        if found_policy:
            print(f"Found existing OPS agent policy: {policy_name}")
            return policy_name
        return None

    @staticmethod
    def create_ops_agent_policy(
        project: str,
        zone: str,
    ) -> None:
        """
        Create an OPS agent policy for the given project and zone.
        See: https://cloud.google.com/stackdriver/docs/solutions/agents/ops-agent/agent-policies-overview
        """
        policy_name = GCPInstanceOpsAgentPolicyCreator.get_policy_ops_agent_policy_name(
            zone
        )
        config_yaml_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        with open(config_yaml_file.name, "w") as f:
            f.write(GCPInstanceOpsAgentPolicyCreator.POLICY)

        print(
            f"Creating OPS agent policy with name: {policy_name} and config:\n{GCPInstanceOpsAgentPolicyCreator.POLICY}"
        )
        policy_cmd = f"""
        gcloud compute instances ops-agents policies create {policy_name} \
            --project={project} \
            --zone={zone} \
            --file={config_yaml_file.name}
        """.strip()
        print(f"Running:\n{policy_cmd}")
        proc = subprocess.run(policy_cmd, shell=True, check=True)
        if proc.returncode != 0:
            print("Error creating OPS agent policy:", proc.stderr)
            raise RuntimeError("Failed to create OPS agent policy.")
        print(f"OPS agent policy '{policy_name}' created.")
        config_yaml_file.close()


if __name__ == "__main__":
    supported_params = SupportedParams()
    parser = argparse.ArgumentParser(
        description="Create a GCP VM instance with the specified parameters."
        " If no parameters are provided, the script will prompt for input."
    )

    for key, param in supported_params.defaults.items():
        help_text = (
            f"{param.description} (default: {param.default})"
            if param.default
            else param.description
        )
        parser.add_argument(f"--{key}", type=str, help=help_text)

    args = parser.parse_args()

    values: Dict[str, str] = {}
    for key, param in supported_params.defaults.items():
        # Check if value is provided in command line arguments
        if getattr(args, key):
            values[key] = getattr(args, key)
            continue
        # If not, prompt for input
        else:
            input_question: str
            long_description_clause = (
                f"\n{param.long_description}" if param.long_description else ""
            )
            if param.default is None:
                required_clause = "(required)" if param.required else "(optional)"
                input_question = f"-> {param.description}{long_description_clause} {required_clause}: "
            else:
                input_question = f"-> {param.description}{long_description_clause}\nDefaults to: [{param.default}]: "

            values[key] = input(input_question).strip() or param.default  # type: ignore
            if not values[key] and param.required:
                raise ValueError(
                    f"Missing required value for {key}. Please provide a value."
                )

    # Check if we need to create the OPS agent policy, and if so try creating it
    skip_install_ops_agent = (
        input(
            "Do you want to install the OPS agent for monitoring and logging? (y/n). Defaults to: [y]: "
        )
        .strip()
        .lower()
        == "n"
    )

    policy_name: Optional[str]
    goog_ops_agent_policy_tag: str = ""
    if not skip_install_ops_agent:
        policy_name = GCPInstanceOpsAgentPolicyCreator.get_existing_ops_agent_policy(
            project=values["project"],
            zone=values["zone"],
        )
        if policy_name:  # If policy already exists this is defined
            should_use_existing_policy = (
                input(
                    f"We found an existing OPS agent policy: {policy_name}; "
                    + "should we re-use it ? (y/n). Defaults to: [y]"
                )
                .strip()
                .lower()
            )
            if should_use_existing_policy == "n":
                print(
                    "Please configure the OPS agent policy manually after instance creation."
                )
                policy_name = None  # We won't configure a policy
            else:
                print(f"Re-using existing OPS agent policy: {policy_name}")
        else:
            print("No existing compatible OPS agent policy found. Creating a new one.")
            policy_name = GCPInstanceOpsAgentPolicyCreator.create_ops_agent_policy(
                project=values["project"],
                zone=values["zone"],
            )

    # The OPS agent policy is
    ops_agent_clause = (
        f",{GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_KEY}={GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_VALUE}"
        if goog_ops_agent_policy_tag
        else ""
    )
    extra_labels_clause = f",{values['labels']}" if values.get("labels") else ""
    startup_script_clause = (
        f",startup-script={values['startup_script']}"
        if values.get("startup_script")
        else ""
    )

    gcloud_cmd = inspect.cleandoc(
        f"""
        gcloud compute instances create {values['machine_name']} \
        --project={values['project']} \
        --zone={values['zone']} \
        --machine-type={values['machine_type']} \
        --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
        --metadata=enable-osconfig=TRUE{startup_script_clause} \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --service-account={values['service_account']} \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --accelerator=count={values['accelerator_count']},type={values['accelerator_type']} \
        --create-disk=auto-delete=yes,boot=yes,device-name={values['machine_name']},image={values['machine_boot_image']},mode=rw,size={values['boot_drive_size_gb']},type=pd-ssd \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=goog-ec-src=vm_add-gcloud{ops_agent_clause}{extra_labels_clause} \
        --reservation-affinity=any
        """
    ).strip()

    print(f"\nRunning gcloud command:\n{gcloud_cmd}")
    result = subprocess.run(gcloud_cmd, shell=True, check=True)
    if result.returncode != 0:
        print("Error creating VM instance:", result.stderr)
        raise RuntimeError("Failed to create VM instance.")
    print("==========================")
    print(
        f"VM instance '{values['machine_name']}' created successfully; you should be able to see it here:\n"
        + f"https://console.cloud.google.com/compute/instancesDetail/zones/{values['zone']}/instances/{values['machine_name']}?project={values['project']}"
    )

    print(f"You should now be able to ssh into your instance.")
    print(
        "Please note when you first ssh into the instance, it may ask for you to install gpu drivers; please do so and reboot the instance so that the OPS agent can work correctly."
    )
