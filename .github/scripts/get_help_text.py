#!/usr/bin/env python3
"""
Script to generate help text for available PR comment commands.
Reads the workflow YAML file and extracts commands with their descriptions.
"""

import os
import re
import textwrap
from pathlib import Path

import yaml


def get_help_text():
    """Generate help text by parsing the workflow file."""
    workflow_path = Path(".github/workflows/on-pr-comment.yml")

    try:
        # Read and parse the YAML file
        with open(workflow_path, "r") as f:
            content = f.read()
            workflow_data = yaml.safe_load(content)

        commands = []

        # Process jobs from parsed YAML
        if "jobs" in workflow_data:
            for job_name, job_config in workflow_data["jobs"].items():
                # Check if this job has a PR comment trigger
                if (
                    "if" in job_config
                    and "contains(github.event.comment.body," in job_config["if"]
                ):
                    # Extract the command from the if condition
                    match = re.search(
                        r"contains\(github\.event\.comment\.body,\s*'([^']+)'\)",
                        job_config["if"],
                    )
                    if match:
                        command = match.group(1)

                        # Get description from the first step's name, or fallback to job name
                        description = f"Run {job_name.replace('-', ' ')} workflow"
                        if (
                            "steps" in job_config
                            and len(job_config["steps"]) > 0
                            and "name" in job_config["steps"][0]
                        ):
                            description = job_config["steps"][0]["name"]

                        commands.append(f"- `{command}` - {description}")

        command_str = "\n".join(commands)
        # Generate the help message
        help_message = textwrap.dedent(
            f"""

        ## 🤖 Available PR Commands

        You can trigger the following workflows by commenting on this PR:

        {command_str}


        💡 **Usage:** Simply comment on this PR with any of the commands above (e.g., `/unit_test`)

        ⏱️ **Note:** Commands may take some time to complete. Progress updates will be posted as comments.
        """
        )

        return help_message

    except Exception as e:
        return f"❌ Error: Could not read workflow file to generate help information. {str(e)}"


def main():
    """Main function to generate help text and set GitHub output."""
    help_text = get_help_text()

    # Set the output for GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            # Use EOF delimiter for multiline output
            f.write(f"help_message<<EOF\n{help_text}\nEOF\n")
    else:
        # Fallback: print to stdout for local testing
        print(help_text)


if __name__ == "__main__":
    main()
