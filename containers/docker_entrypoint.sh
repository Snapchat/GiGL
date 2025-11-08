#!/bin/bash
source /tmp/.venv/bin/activate

# Pass command arguments to the default beam boot script.
/opt/apache/beam/boot "$@"