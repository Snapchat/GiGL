#!/bin/bash
# This script is used to download and install the scala dependencies for the project.
# The script exposes a --download-only flag to only download the files, and skip the installation and setup.
# This is useful when building a docker image, for when we want the host context to have all the scala deps
# so we can copy over the deps from the host into the built docker image, without mounting and credentials
# to the docker build - which can be prone to danger. Currently, our hosted GCS assets have:
# - allAuthenticatedUsers read access, thus one needs to be authenticated w/ GCP to download the files.

set -e
set -x

# Parse command line arguments
DOWNLOAD_ONLY=false
for arg in "$@"; do
    case $arg in
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--download-only]"
            echo "  --download-only: Only download files, skip installation and setup"
            exit 1
            ;;
    esac
done

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"
# Source the dep_vars.env file to use its variables
source "$SCRIPT_DIR/../gigl/dep_vars.env"

# Local path variables
TOOLS_SCALA_DIR="tools/scala"
TOOLS_SCALAPBC_DIR="tools/scalapbc"
TOOLS_COURSIER_DIR="tools/scala/coursier"

COURSIER_GZ_LOCAL_PATH="$TOOLS_COURSIER_DIR/cs-x86_64-pc-linux.gz"
COURSIER_BINARY_LOCAL_PATH="$TOOLS_COURSIER_DIR/cs"
SPARK_31_TAR_LOCAL_PATH="$TOOLS_SCALA_DIR/spark-3.1.3-bin-hadoop3.2.tgz"
SPARK_35_TAR_LOCAL_PATH="$TOOLS_SCALA_DIR/spark-3.5.0-bin-hadoop3.tgz"
SCALAPB_0_11_11_LOCAL_PATH="$TOOLS_SCALAPBC_DIR/scalapbc-0.11.11.zip"
SCALAPB_0_11_14_LOCAL_PATH="$TOOLS_SCALAPBC_DIR/scalapbc-0.11.14.zip"

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

# Function to download file only if it doesn't exist locally
download_if_missing() {
    local gcs_path="$1"
    local local_path="$2"

    if [ -f "$local_path" ]; then
        echo "File already exists locally: $local_path, skipping download"
    else
        echo "Downloading $gcs_path to $local_path"
        gsutil cp "$gcs_path" "$local_path"
    fi
}


download_all_scala_deps() {
    echo "Creating required directories"
    mkdir -p "$TOOLS_COURSIER_DIR"
    mkdir -p "$TOOLS_SCALAPBC_DIR"
    mkdir -p "$TOOLS_SCALA_DIR"

    echo "Downloading all required files..."

    # Download coursier for Linux (Mac uses brew), or if running in download-only mode
    if ! is_running_on_mac || [ "$DOWNLOAD_ONLY" = true ]; then
        download_if_missing "gs://public-gigl/tools/scala/coursier/cs-x86_64-pc-linux.gz" "$COURSIER_GZ_LOCAL_PATH"
    fi

    # Download Spark distributions
    download_if_missing "gs://public-gigl/tools/scala/spark/spark-3.1.3-bin-hadoop3.2.tgz" "$SPARK_31_TAR_LOCAL_PATH"
    download_if_missing "gs://public-gigl/tools/scala/spark/spark-3.5.0-bin-hadoop3.tgz" "$SPARK_35_TAR_LOCAL_PATH"

    # Download TFRecord JARs
    download_if_missing "$SPARK_31_TFRECORD_JAR_GCS_PATH" "$SPARK_31_TFRECORD_JAR_LOCAL_PATH"
    download_if_missing "$SPARK_35_TFRECORD_JAR_GCS_PATH" "$SPARK_35_TFRECORD_JAR_LOCAL_PATH"

    # Download ScalaPB distributions
    download_if_missing "gs://public-gigl/tools/scala/scalapbc/scalapbc-0.11.11.zip" "$SCALAPB_0_11_11_LOCAL_PATH"
    download_if_missing "gs://public-gigl/tools/scala/scalapbc/scalapbc-0.11.14.zip" "$SCALAPB_0_11_14_LOCAL_PATH"

    echo "All downloads completed successfully"
}

download_all_scala_deps
# Exit if download-only flag is set
if [ "$DOWNLOAD_ONLY" = true ]; then
    echo "Download-only mode: Skipping installation and setup steps"
    exit 0
fi

# Now proceed with installations and setup
if is_running_on_mac;
then
    echo "Setting up Scala Deps for Mac environment"
    brew install coursier/formulas/coursier && cs setup
    brew install sbt
else
    echo "Setting up Scala Deps for Linux Environment"
    gunzip -c "$COURSIER_GZ_LOCAL_PATH" > "$COURSIER_BINARY_LOCAL_PATH" && chmod +x "$COURSIER_BINARY_LOCAL_PATH" && "$COURSIER_BINARY_LOCAL_PATH" setup -y
    rm "$COURSIER_GZ_LOCAL_PATH"
fi

source ~/.profile

echo "Setting up required tooling"

echo "Installing tooling for spark 3.1"
gunzip -c "$SPARK_31_TAR_LOCAL_PATH" | tar xopf - -C "$TOOLS_SCALA_DIR"

echo "Installing tooling for spark 3.5; this will deprecate regular installation for spark 3.1 above"
gunzip -c "$SPARK_35_TAR_LOCAL_PATH" | tar xopf - -C "$TOOLS_SCALA_DIR"

# Remove SparkR, as it contains GPL licensed code - even though Spark is Apache licensed
# We don't use this anyways.
rm -rf "$TOOLS_SCALA_DIR/spark-3.1.3-bin-hadoop3.2/R"
rm -rf "$TOOLS_SCALA_DIR/spark-3.5.0-bin-hadoop3/R"


echo "Installing tooling for scala protobuf"
unzip -o "$SCALAPB_0_11_11_LOCAL_PATH" -d "$TOOLS_SCALAPBC_DIR"
rm "$SCALAPB_0_11_11_LOCAL_PATH"
# (svij-sc) scala35 support (this will eventually deprecate above)
unzip -o "$SCALAPB_0_11_14_LOCAL_PATH" -d "$TOOLS_SCALAPBC_DIR"
rm "$SCALAPB_0_11_14_LOCAL_PATH"

# if running into the following error when running unittest locally
# `java.lang.NoClassDefFoundError: Could not initialize class org.apache.spark.storage.StorageUtils$`
# export JAVA_OPTS='--add-exports java.base/sun.nio.ch=ALL-UNNAMED'
echo "Finished Scala environment installation"
