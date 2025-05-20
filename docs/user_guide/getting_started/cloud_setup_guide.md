# Cloud Setup Guide

```{note}
The guidance below assumes you are not operating within a corporate network or under organization-specific policies. It provides instructions for onboarding directly onto raw GCP.

If your company or lab uses custom IAM roles, security policies, infrastructure management tools, or other internal systems, please refer to your internal documentation for how those may apply alongside the steps outlined here.

For more detailed information on meeting the prerequisites, refer to the official [GCP documentation](https://cloud.google.com/docs).
```

## GCP Project Setup Guide

1. A GCP account with billing enabled.

2. Created a GCP project.

3. Get `roles/editor` access to the GCP project

4. [Install and setup the gcloud CLI on your local machine](https://cloud.google.com/sdk/docs/install)

5. Enabled the necessary APIs on your GCP project:

   - [Compute Engine](https://console.cloud.google.com/apis/library/compute.googleapis.com)
   - [Dataflow](https://console.cloud.google.com/apis/library/dataflow.googleapis.com)
   - [VertexAI](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
   - [Dataproc](https://console.cloud.google.com/apis/library/dataproc.googleapis.com)
   - [BigQuery](https://console.cloud.google.com/apis/library/bigquery.googleapis.com)
   - [Artifact Registry](https://console.cloud.google.com/apis/library/artifactregistry.googleapis.com)

6. [Created GCS bucket(s)](https://console.cloud.google.com/storage/create-bucket) for storing assets. You need to
   create two different buckets for storing temporary and permanent assets. We will reference these as
   `temp_assets_bucket`, and `perm_assets_bucket` respectively throughout the library.

   - Pro-tip: Create regional buckets and use the same region for all your resources and compute to keep your cloud
     costs minimal i.e. `us-central1`
     <img src="../../assets/images/cloud_setup/regional_bucket_example.png" alt="Regional Bucket Example" width="500px" />
   - Ensure to use the "standard" default class for storage
   - (Optional) Enable Hierarchical namespace on this bucket
   - For your temp bucket its okay to disable `Soft delete policy (For data recovery)`, otherwise you will get billed
     unecessarily for large armounts of intermediary assets GiGL creates.
   - Since GiGL creates alot of intermediary assets you will want to create a
     [lifecycle rule](https://cloud.google.com/storage/docs/lifecycle) on the temporariy bucket to automatically delete
     assets. Intermediary assets can add up very quickly - you have been warned. Example:
     <img src="../../assets/images/cloud_setup/bucket_lifecycle_example.png" alt="Bucket Lifecycle Example"/>

7. [Create BQ Datasets](https://console.cloud.google.com/bigquery) for storing assets. You need to create two different
   BQ datasets in the project, one for storing temporary assets, and one for output embeddings. We will reference these
   as `temp_assets_bq_dataset_name`, and `embedding_bq_dataset_name` respectively throughout the library.

   ```{caution}
   The BQ datasets must be in the same project and be multi regional with the location being the country/superset region where you plan on running the pipelines - otherwise pipelines won't work. i.e. `us-central1` region maps to `US` for multi_regional dataset.
   <img src="../../assets/images/cloud_setup/multi_regional_bq_dataset_example.png" alt="Regional BQ Dataset Example">
   ```

8. [Create a Docker Artifact Registry](https://console.cloud.google.com/artifacts) for storing your compiled docker
   images that will contain your custom source code GiGL source. Ensure the registry is in the same region as your other
   compute assets.

   <img src="../../assets/images/cloud_setup/regional_registry_example.png" alt="Regional Registry Example">

9. Create a [new GCP service account](https://console.cloud.google.com/iam-admin/serviceaccounts) (or use an existing),
   and [give it relevant IAM perms](https://cloud.google.com/iam/docs/roles-overview):

```{note}
  You youself are going to need the following permissions to create new IAM bindings: `roles/resourcemanager.projectIamAdmin`, and `roles/iam.serviceAccountAdmin`
```

a. Firstly give the SA permission to use itself:

```bash
 gcloud iam service-accounts add-iam-policy-binding $SERVICE_ACCOUNT \
 --member="serviceAccount:$SERVICE_ACCOUNT" \
 --role="roles/iam.serviceAccountUser"
```

b. Next, the SA will need some permissions at the project level:

````{note}
Example of granting `bigquery.user`:
  ```bash
    gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/bigquery.user"
  ```
````

- `bigquery.user`
- `cloudprofiler.user`
- `compute.admin`
- `dataflow.admin`
- `dataflow.worker`
- `dataproc.editor`
- `logging.logWriter`
- `monitoring.metricWriter`
- `notebooks.legacyViewer`
- `aiplatform.user`
- `dataproc.worker`
- `artifactregistry.writer`

c. Next we need to grant the GCP
[Vertex AI Service Agent](https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents) i.e.
`service-$PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com` permissions to read from artifact registry so the
VAI pipelines can pull the docker images you push to the registry you created. You can find your `$PROJECT_NUMBER` from
your main project console page: `console.cloud.google.com`

```bash
 gcloud projects add-iam-policy-binding $PROJECT_ID \
   --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
   --role="roles/artifactregistry.reader"
```

10. Give your SA `storage.objectAdmin` and `roles/storage.legacyBucketReader` on the buckets you created i.e. to grant
    `storage.objectAdmin` you can run:

```bash
gcloud storage buckets add-iam-policy-binding $BUCKET_NAME \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectAdmin"
```

11. Give your SA `roles/bigquery.dataOwner` on the datasets you created. See
    [instructions](https://cloud.google.com/bigquery/docs/control-access-to-resources-iam#bq_2).

## AWS Project Setup Guide

- Not yet supported
