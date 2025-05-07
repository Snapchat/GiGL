# Cloud Setup Guide

## GCP Project Setup Guide

- A GCP account with billing enabled.
- Created a GCP project.
- Enabled the necessary APIs:
  - Compute Engine
  - Dataflow
  - VertexAI
  - Dataproc
  - BigQuery
- Set up a GCP service account, and give it relevant perms:
  - bigquery.user
  - cloudprofiler.user
  - compute.admin
  - dataflow.admin
  - dataflow.worker
  - dataproc.editor
  - logging.logWriter
  - monitoring.metricWriter
  - notebooks.legacyViewer
  - aiplatform.user
  - dataproc.worker
- Created a GCS bucket(s) for storing assets. You can specify two different buckets for storing temporary and permanent
  assets. At large scale GiGL creates alot of intermediary assets; so you may want to create a bucket for storing these
  temp assets and set a lifecycle rule on it to automatically delete assets.
  - Give your service account storage.objectAdmin perms for the bucket(s) you created

Refer to the [GCP documentation](https://cloud.google.com/docs) for detailed instructions on meeting these
prerequisites.

## AWS Project Setup Guide

- TODO (Not yet supported)
