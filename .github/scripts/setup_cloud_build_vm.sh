#!/bin/bash
set -e
set -x

source ~/.bashrc

# Configure docker for gcloud.
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
gcloud auth list
gcloud config list project
