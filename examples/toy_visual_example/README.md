## Setup Resource Config.

GiGL requires you to setup a [resource config](../../../docs/user_guide/config_guides/resource_config_guide.md) in order
to launch jobs on GCP. There is already [resource_config.yaml](./resource_config.yaml) in this directory but it is setup
for GiGL CICD, and you will not be able to use it.

You may run the below from GiGL root to generate an appropriate resource config. If you are onboarding to GiGL through
qwiklabs then you can directly use the below commands. If not, then you will need to setup your GCP project per our
[cloud setup guide](../../docs/user_guide/getting_started/cloud_setup_guide.md) before running the below command with
the appropriate resources.

```bash
PROJECT="$(gcloud config get-value project)" # Ex, qwiklabs-gcp-01-40f6ccb540f3
QL_USER=$QWIK_LABS_USER # Ex, student-02-5e0049fb83ce

python -m scripts.bootstrap_resource_config \
  --project="$PROJECT" \
  --gcp_service_account_email="gigl-dev@$PROJECT.iam.gserviceaccount.com" \
  --docker_artifact_registry_path="us-central1-docker.pkg.dev/$PROJECT/gigl-base-images" \
  --temp_assets_bq_dataset_name="gigl_temp_assets" \
  --embedding_bq_dataset_name="gigl_embeddings" \
  --temp_assets_bucket="gs://gigl_temp_assets_$QL_USER" \
  --perm_assets_bucket="gs://gigl_perm_assets_$QL_USER" \
  --template_resource_config_uri="examples/toy_visual_example/resource_config.yaml"
```

Accept the default region `us-central1` and use the default file output location.
