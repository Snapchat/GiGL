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


class GCPInstanceOpsAgentPolicyCreator:
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
        Get the ops agent policy name for the given zone.
        """
        return f"goog-ops-agent-v2-x86-template-1-4-0-{zone}"

    @staticmethod
    def get_existing_ops_agent_policy(project: str, zone: str) -> Optional[str]:
        """
        Get the existing ops agent policy for the given project and zone.
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
            print("Ops agent policy not found or error retrieving:", e)

        if found_policy:
            print(f"Found existing ops agent policy: {policy_name}")
            return policy_name
        return None

    @staticmethod
    def create_ops_agent_policy(
        project: str,
        zone: str,
    ) -> None:
        """
        Create an ops agent policy for the given project and zone.
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
            f"Creating ops agent policy with name: {policy_name} and config:\n{GCPInstanceOpsAgentPolicyCreator.POLICY}"
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
            print("Error creating ops agent policy:", proc.stderr)
            raise RuntimeError("Failed to create ops agent policy.")
        print(f"Ops agent policy '{policy_name}' created.")
        config_yaml_file.close()


if __name__ == "__main__":
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

    defaults: Dict[str, Param] = {
        "PROJECT": Param(default=project, description="GCP project ID"),
        "ZONE": Param(default="us-central1-a", description="GCP zone"),
        "SERVICE_ACCOUNT": Param(default=None, description="Service account email"),
        "MACHINE_NAME": Param(
            default=f"{whoami}-gigl-dev-instance", description="Name of the VM instance"
        ),
        "MACHINE_TYPE": Param(default="n1-highmem-32", description="Machine type"),
        "ACCELERATOR_TYPE": Param(
            default="nvidia-tesla-t4", description="GPU accelerator type"
        ),
        "ACCELERATOR_COUNT": Param(
            default="4", description="Number of GPUs to attach to the VM"
        ),
        "MACHINE_BOOT_IMAGE": Param(
            default="projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda",
            description="Boot image for the VM.",
            long_description="You can find images google provides @ https://cloud.google.com/deep-learning-vm/docs/images. \n"
            + "Please ensure to use an image whose cuda version is compatible with GiGL.\n"
            + "See the following for GiGL supported cuda versions: "
            + "https://snapchat.github.io/GiGL/docs/user_guide/getting_started/installation.html#supported-environments",
        ),
        "BOOT_DRIVE_SIZE_GB": Param(default="1000", description="Boot disk size in GB"),
        "STARTUP_SCRIPT": Param(
            default=None,
            required=False,
            description="Startup script to run on instance creation",
            long_description="This is a bash script that will be run on instance creation. "
            + "It is recommended to use this to install any dependencies you need for your GiGL instance.",
        ),
    }

    values: dict[str, str] = {}
    for key, param in defaults.items():
        input_question: str
        long_description_clause = (
            f"\n{param.long_description}" if param.long_description else ""
        )
        if param.default is None:
            input_question = (
                f"-> {param.description}{long_description_clause} (required): "
            )
        else:
            input_question = f"-> {param.description}{long_description_clause}\nDefaults to: [{param.default}]: "

        values[key] = input(input_question).strip() or param.default  # type: ignore
        if not values[key] and param.required:
            raise ValueError(
                f"Missing required value for {key}. Please provide a value."
            )

    # Check if we need to create the ops agent policy, and if so try creating it
    skip_install_ops_agent = (
        input(
            "Do you want to install the ops agent for monitoring and logging? (y/n). Defaults to: [y]: "
        )
        .strip()
        .lower()
        == "n"
    )

    policy_name: Optional[str]
    goog_ops_agent_policy_tag: str = ""
    if not skip_install_ops_agent:
        policy_name = GCPInstanceOpsAgentPolicyCreator.get_existing_ops_agent_policy(
            project=values["PROJECT"],
            zone=values["ZONE"],
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
                    "Please configure the ops agent policy manually after instance creation."
                )
                policy_name = None  # We won't configure a policy
            else:
                print(f"Re-using existing ops agent policy: {policy_name}")
        else:
            print("No existing compatible ops agent policy found. Creating a new one.")
            policy_name = GCPInstanceOpsAgentPolicyCreator.create_ops_agent_policy(
                project=values["PROJECT"],
                zone=values["ZONE"],
            )

    ops_agent_clause = (
        f",{GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_KEY}={GCPInstanceOpsAgentPolicyCreator.POLICY_INCLUSION_LABEL_VALUE}"
        if goog_ops_agent_policy_tag
        else ""
    )
    startup_script_clause = (
        f",startup-script={values['STARTUP_SCRIPT']}"
        if values.get("STARTUP_SCRIPT")
        else ""
    )

    gcloud_cmd = inspect.cleandoc(
        f"""
    gcloud compute instances create {values['MACHINE_NAME']} \
    --project={values['PROJECT']} \
    --zone={values['ZONE']} \
    --machine-type={values['MACHINE_TYPE']} \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --metadata=enable-osconfig=TRUE{startup_script_clause} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account={values['SERVICE_ACCOUNT']} \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --accelerator=count={values['ACCELERATOR_COUNT']},type={values['ACCELERATOR_TYPE']} \
    --create-disk=auto-delete=yes,boot=yes,device-name={values['MACHINE_NAME']},image={values['MACHINE_BOOT_IMAGE']},mode=rw,size={values['BOOT_DRIVE_SIZE_GB']},type=pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud{ops_agent_clause} \
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
        f"VM instance '{values['MACHINE_NAME']}' created successfully; you should be able to see it here:\n"
        + f"https://console.cloud.google.com/compute/instancesDetail/zones/{values['ZONE']}/instances/{values['MACHINE_NAME']}?project={values['PROJECT']}"
    )

    print(f"You should now be able to ssh into your instance.")
    print(
        "Please note when you first ssh into the instance, it may ask for you to install gpu drivers; please do so and reboot the instance so that the OPS agent can work correctly."
    )
