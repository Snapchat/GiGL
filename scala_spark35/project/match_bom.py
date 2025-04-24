# Why is this needed?
# To make sure that we have compatible versions of Cloud Client Libraries,
# we use the versions specified in the Google Cloud Libraries Bill of Materials (BOM). 
# The libraries in the BOM don't have dependency conflicts that would manifest as 
# NoSuchMethodError or NoClassDefFoundError.
# See https://cloud.google.com/java/docs/bom for more information.

#  The current version of BOM used: 26.2.0
#  https://storage.googleapis.com/cloud-opensource-java-dashboard/com.google.cloud/libraries-bom/26.2.0/index.html
# To generate the list of dependencies for the BOM, do the following:
# 0. Update the BOM_VERSION in the script below
# 1. Open build.sbt and comment out the line `) ++ GCPDepsBOM.v26_2_0_deps` --> `) // ++ GCPDepsBOM.v26_2_0_deps`
# 2. Run the following command: python project/match_bom.py
# 3. Copy the output and paste it into the file project/GCPDepsBOM.scala
# 4. Uncomment the line in build.sbt

from typing import NewType, Dict, Tuple
import subprocess

BOM_VERSION = "26.2.0"

# Assuming GiGLComponents is an enum or a class and you want a specific type for job name suffix
Group = NewType('Group', str)
Artifact = NewType('Artifact', str)
Version = NewType('Version', str)

GREP_REGEX_GOOGLE_DEPS = "grep -o 'com\.google\.[a-zA-Z0-9_.-]*.[a-zA-Z0-9_.-]*.[a-zA-Z0-9_.-]*'"
URL_BASE_BOM_DEPS: str = "https://storage.googleapis.com/cloud-opensource-java-dashboard/com.google.cloud/libraries-bom/"


def run_command(cmd: str) -> str:
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print("Command executed successfully")
    else:
        print(f"Command failed with return code {result.returncode}")
        print(result.stdout)
        print(result.stderr)
        exit(1)
    return result.stdout

def get_deps_dict(deps: str) -> Dict[Tuple[Group, Artifact], Version]:
    deps_dict: Dict[Tuple[Group, Artifact], Version] = {}
    for line in deps.splitlines():
        group, artifact, version = line.split(':')
        deps_dict[(Group(group), Artifact(artifact))] = Version(version)
    return deps_dict

def get_bom_deps(bom_version: Version) -> Dict[Tuple[Group, Artifact], Version]:
    print(f"Getting dependencies for BOM version {bom_version}")
    url_bom_deps_versioned = f"{URL_BASE_BOM_DEPS}{bom_version}/index.html"
    cmd_get_bom_deps = f"curl '{url_bom_deps_versioned}' | {GREP_REGEX_GOOGLE_DEPS}"
    result_stdout = run_command(cmd_get_bom_deps)
    return get_deps_dict(result_stdout)


def get_current_deps() -> Dict[Tuple[Group, Artifact], Version]:
    cmd_get_current_deps = f"sbt dependencyTree | {GREP_REGEX_GOOGLE_DEPS}"
    result_stdout = run_command(cmd_get_current_deps)
    return get_deps_dict(result_stdout)

        

if __name__ == "__main__":
    bom_deps = get_bom_deps(Version(BOM_VERSION))
    current_deps = get_current_deps()
    # Now map the current deps to have the version from the BOM
    # Note: We don't want to use all the BOM libraries as most of them are not needed and woould
    # just inflate our jar size.
    new_deps = {}
    for key in current_deps.keys():
        if key in bom_deps:
            new_deps[key] = bom_deps[key]
        else:
            print(f"WARNING: {key} not found in BOM")
    
    # Print the new deps so they can be copied into GCPDepsBOM.scala
    for key, value in new_deps.items():
        print(f"        \"{key[0]}\" % \"{key[1]}\" % \"{value}\",")