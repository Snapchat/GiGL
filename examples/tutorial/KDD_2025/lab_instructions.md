# Training Industry-scale GNNs with GiGL: KDD'25 Hands-On Tutorial (attendee)

## Introduction

Thanks for your interest and welcome to the KDD'25 tutorial on Training Industry-scale GNNs with GiGL! This tutorial
aims to provide a comprehensive introduction to Graph Neural Networks (GNNs) and their scalability challenges, while
also offering hands-on experience with the GiGL library. By the end of this tutorial, you will have a solid
understanding of GNNs, the challenges they face in scaling, and how GiGL addresses these challenges effectively. Most
importantly, you will walk away with the knowledge and skills to train and infer industry-scale GNNs using GiGL.

Thinking about integrating Graph Neural Networks into your applications? Here's why GiGL stands out:

- **Built for scale**: GiGL is a large-scale Graph Neural Network library that works on industry-scale graphs.
- **Efficient and cost-effective**: GiGL addresses GNN scalability challenges and provide cost-effective solutions for
  training and inference.
- **Easy to adopt**: GiGL has abstracted interfaces and compatibility with popular GNN frameworks like PyG.
- **Battle-tested at Snapchat**: GiGL is widely used at Snapchat and has been successfully deployed in production for
  various applications.

## What you'll learn

Hands-On experience with the GiGL library to train industry-scale Graph Neural Networks (GNNs):

- How to use the various API components that GiGL provides to process and train your large-scale graph.
- How to perform end-to-end GNN training with GiGL in distributed environments.
- How to customize your GNN training pipeline using GiGL.

## Lab Instructions

1. Click start lab on the left side of the page. Wait about 4 minutes for the lab to be set up.
2. Right click on `Open Google Cloud Console` button and open in a incognito window.
3. Use the top search bar to navigate to the `Workbench` page provided by `Vertex AI`.
4. There should be an existing Workbench instance, open jupyterlab.
5. Open the `GiGL` folder in the left sidebar. Navigate to the `examples/tutorial/KDD_2025` folder.

## Setup Resource Config.

The tutorial requires you to setup a [resource config](../../../docs/user_guide/config_guides/resource_config_guide.md)
in order to launch jobs on GCP.

You may run the below from GiGL root to generate an appropriate resource config. You may find the Project and User on
the left hand panel on the qwiklabs page.

```bash
PROJECT=$QWIK_LABS_PROJECT # Ex, qwiklabs-gcp-01-40f6ccb540f3
QL_USER=$QWIK_LABS_USER # Ex, student-02-5e0049fb83ce

python -m scripts.bootstrap_resource_config \
  --project=$PROJECT \
  --gcp_service_account_email=gigl-dev@$PROJECT.iam.gserviceaccount.com \
  --docker_artifact_registry_path=us-central1-docker.pkg.dev/$PROJECT/gigl-images \
  --temp_assets_bq_dataset_name="gigl_temp_assets" \
  --embedding_bq_dataset_name="gigl_temp_assets" \
  --temp_assets_bucket="gs://gigl_temp_assets_$QL_USER" \
  --perm_assets_bucket="gs://gigl_perm_assets_$QL_USER"
```

Accept the default region `us-central1` and output the resource config somewhere locally, like
`examples/tutorial/KDD_2025/resource_config.yaml`.

## Additional Resources

- GiGL KDD ADS track paper: [Paper link](https://arxiv.org/abs/2502.15054)
- GiGL library source code: [GitHub](https://github.com/Snapchat/GiGL/tree/main)
- GiGL documentation: [Read the Docs](https://snapchat.github.io/GiGL/index.html)
