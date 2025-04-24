from pathlib import Path

DEP_VARS_FILE_PATH = Path.joinpath(Path(__file__).parent.parent.parent, "dep_vars.env")

if __name__ == "__main__":
    assert (
        DEP_VARS_FILE_PATH.exists()
    ), f"File `dep_vars.env` not found at: {DEP_VARS_FILE_PATH}"
    with open(file=DEP_VARS_FILE_PATH, mode="r") as f:
        # Ensure we only have comments, empty lines, or lines with variable definitions
        for line in f.readlines():
            if line.startswith("#") or not line.strip():  # Is line a comment or empty?
                continue
            if (
                "=" not in line or ":=" in line
            ):  # := dictates runtime evaluation of the variable; = is static
                raise ValueError(
                    f"Invalid line found in `dep_vars.env`: {line}. Expected format: var=value"
                )
