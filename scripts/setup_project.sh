#!/bin/bash

set -euo pipefail
trap 'echo "❌ Script failed on line $LINENO"; exit 1' ERR

if [ $# -lt 1 ]; then
  echo "Usage: $0 <user_email>"
  exit 1
fi

# === Inputs & Derived Values ===
USER_EMAIL="$1"
USER_NAME="${USER_EMAIL%@*}"
PROJECT_ID="$(gcloud config get-value project)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")"
REGION="us-central1"
LOCATION="US"
SA_NAME="gigl-dev"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# === Resources ===
GCS_BUCKETS=(gigl_temp_assets gigl_perm_assets)
BQ_DATASETS=(gigl_temp_assets gigl_embeddings)
PROJECT_ROLES=(
  roles/editor
  roles/iam.serviceAccountAdmin
  roles/resourcemanager.projectIamAdmin
)
SA_PROJECT_ROLES=(
  roles/bigquery.user
  roles/cloudprofiler.user
  roles/compute.admin
  roles/dataflow.admin
  roles/dataflow.worker
  roles/dataproc.editor
  roles/logging.logWriter
  roles/monitoring.metricWriter
  roles/notebooks.legacyViewer
  roles/aiplatform.user
  roles/dataproc.worker
  roles/artifactregistry.writer
)
APIS=(
  compute.googleapis.com
  dataflow.googleapis.com
  aiplatform.googleapis.com
  dataproc.googleapis.com
  bigquery.googleapis.com
  artifactregistry.googleapis.com
  osconfig.googleapis.com
)

echo "=== Project: $PROJECT_ID"
echo "=== Region: $REGION"
echo "=== User: $USER_EMAIL"
echo "=== Service Account: $SA_EMAIL"
echo "=== Project Number: $PROJECT_NUMBER"


# === Grant user roles ===
for ROLE in "${PROJECT_ROLES[@]}"; do
  echo "Granting $ROLE to $USER_EMAIL"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="user:$USER_EMAIL" \
    --role="$ROLE"
done

# === Enable required APIs ===
echo "Enabling required APIs..."
for API in "${APIS[@]}"; do
  echo "→ Enabling $API"
  gcloud services enable "$API" --project="$PROJECT_ID"
done


# === Create GCS buckets ===
for BUCKET in "${GCS_BUCKETS[@]}"; do
  echo "Creating GCS bucket: $BUCKET"
  gsutil mb -c standard -l "$LOCATION" -p "$PROJECT_ID" "gs://${BUCKET}_${USER_NAME}/"
done

# === Create BigQuery datasets ===
for DATASET in "${BQ_DATASETS[@]}"; do
  echo "Creating BigQuery dataset: $DATASET"
  bq --location=US mk --dataset "${PROJECT_ID}:${DATASET}" || true
done

# === Create Artifact Registry ===
echo "Creating Artifact Registry: gigl-base-images"
gcloud artifacts repositories create gigl-base-images \
  --repository-format=docker \
  --location="$REGION" \
  --description="Base Docker images for GIGL" || true

# === Create service account ===
echo "Creating service account: $SA_EMAIL"
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="GIGL Dev Service Account" || true

sleep 5

# === Allow the SA to use itself ===
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"

# === Grant project-level roles to SA ===
for ROLE in "${SA_PROJECT_ROLES[@]}"; do
  echo "Granting $ROLE to $SA_EMAIL"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="$ROLE"
done

# === Grant SA permissions on GCS buckets ===
for BUCKET in "${GCS_BUCKETS[@]}"; do
  echo "Granting GCS roles on $BUCKET to $SA_EMAIL"
  gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET}_${USER_NAME}"
  gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.legacyBucketReader" "gs://${BUCKET}_${USER_NAME}"
done

# === Grant SA permissions on BQ datasets ===
for DATASET in "${BQ_DATASETS[@]}"; do
  echo "Granting BigQuery OWNER role on $DATASET to $SA_EMAIL"
  bq show --format=prettyjson "$PROJECT_ID:$DATASET" > /tmp/ds.json
  jq --arg sa "$SA_EMAIL" '.access += [{"role":"OWNER","userByEmail":$sa}]' /tmp/ds.json > /tmp/ds_new.json
  bq update --source /tmp/ds_new.json "$PROJECT_ID:$DATASET"
done

# === Grant Vertex AI service agent Artifact Registry reader role ===

echo "Ensuring Vertex AI Custom Code Service Agent exists..."

# The agent is only created when custom-code is run: see https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents.
# Run a dummy custom job to trigger agent creation.
gcloud ai custom-jobs create \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --display-name="probe-agent" \
  --worker-pool-spec=replica-count=1,machine-type=n1-highmem-2,container-image-uri=gcr.io/deeplearning-platform-release/base-cpu \
  --command="sleep" \
  --args="10" \
  --quiet >/dev/null 2>&1 || true

# Wait for the agent to appear (poll for up to 5 minutes)
VERTEX_AGENT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

echo "Attempting to bind Artifact Registry reader role to Vertex AI service agent: $VERTEX_AGENT"

for i in {1..30}; do
  if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${VERTEX_AGENT}" \
    --role="roles/artifactregistry.reader" \
    --quiet >/dev/null 2>&1; then
    echo "Successfully bound Artifact Registry reader role to $VERTEX_AGENT"
    break
  else
    echo "Attempt $i failed — vai service account may not exist yet. Retrying in 10 seconds..."
    sleep 10
  fi
done

# === Grant Dataflow default account GiGL SA user permission

gcloud dataflow jobs run test-word-count \
    --gcs-location=gs://dataflow-templates/latest/Word_Count \
    --region="${REGION}" \
    --parameters="inputFile=gs://dataflow-samples/shakespeare/kinglear.txt,output=gs://word_count/output/wordcount_results"

echo "Attempting to bind GiGL service account user role to Dataflow default account"
for i in {1..30}; do
  if gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --member="serviceAccount:service-${PROJECT_NUMBER}@dataflow-service-producer-prod.iam.gserviceaccount.com"\
    --role="roles/iam.serviceAccountUser"; then
    break
  else
    echo "Attempt $i failed — dataflow service account may not exist yet. Retrying in 10 seconds..."
    sleep 10
  fi
done


# === Grant Dataproc account GiGL SA user permission
# Start dataproc cluster to activate dataproc SA, move on regardless of command succesful or not
gcloud dataproc clusters create my-test-cluster-1 \
    --region=us-central1 \
    --service-account="${SA_EMAIL}" || true

for i in {1..30}; do
  if gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
      --member="serviceAccount:service-${PROJECT_NUMBER}@dataproc-accounts.iam.gserviceaccount.com" \
      --role="roles/iam.serviceAccountUser"; then
      break
  else
    echo "Attempt $i failed — dataproc service account may not exist yet. Retrying in 10 seconds..."
    sleep 10
  fi
done


echo ""
echo "=== Created Resources ==="
echo "Project ID: $PROJECT_ID"
echo "Service Account: $SA_EMAIL"
echo "GCS Buckets:"
for BUCKET in "${GCS_BUCKETS[@]}"; do
  echo "  - gs://${BUCKET}_${USER_NAME}/"
done
echo "BigQuery Datasets:"
for DATASET in "${BQ_DATASETS[@]}"; do
  echo "  - ${PROJECT_ID}:${DATASET}"
done
echo ""
