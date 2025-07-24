# <p align="center"> Training Industry-scale GNNs with GiGL: KDD'25 Hands-On Tutorial </p>

______________________________________________________________________

## <p align="center">**To appear at KDD'25 - Stay Tuned**</p>

______________________________________________________________________

## Tutorial Goal

Thanks for your interest and welcome to the KDD'25 tutorial on **Training Industry-scale GNNs with GiGL**! This tutorial
aims to provide a comprehensive introduction to Graph Neural Networks (GNNs) and their scalability challenges, while
also offering hands-on experience with the GiGL library. By the end of this tutorial, you will have a solid
understanding of GNNs, the challenges they face in scaling, and how GiGL addresses these challenges effectively. Most
importantly, you will walk away with the knowledge and skills to train and infer industry-scale GNNs using GiGL.

Thinking about integrating Graph Neural Networks into your applications? Here's why GiGL stands out:

- **Built for scale:** GiGL is a large-scale Graph Neural Network library that works on industry-scale graphs.
- **Efficient and cost-effective:** GiGL addresses GNN scalability challenges and provide cost-effective solutions for
  training and inference.
- **Easy to adopt:** GiGL has abstracted interfaces and compatibility with popular GNN frameworks like PyG.
- **Battle-tested at Snapchat:** GiGL is widely used at Snapchat and has been successfully deployed in production for
  various applications.

## Tutorial Outline

### GNNs and their Scale Challenges (20m)

- **GNN Fundamentals and Formalisms:** This section will cover the basics of Graph Neural Networks, including their
  underlying mathematical concepts and how they operate.
- **Real-world Applications:** We'll explore various practical applications of GNNs across different industries,
  showcasing their versatility and impact.
- **Scalability Challenges:** This part will delve into the inherent difficulties of scaling GNNs to handle large,
  real-world datasets, highlighting the computational and memory limitations.

### Overview of GIGL (20m)

- **Technical Scaling Strategy:** We'll discuss the technical approaches and strategies employed by GIGL to overcome the
  scalability challenges of GNNs.
- **Library Design:** This section will provide an overview of GIGL's architectural design, explaining how its
  components are structured and interact.
- **GIGL Core Components:** We'll identify and describe the key building blocks and functionalities that make up the
  GIGL library.

### **Hands-on with GIGL**: Training and Inferring Industry-Scale GNNs - Part I (30m)

- **Environment Setup:** This hands-on segment will guide you through setting up the necessary software and hardware
  environment to work with GIGL.
- **Provisioning Access to Large-Scale Graph Data:** We'll cover how to access and prepare large-scale graph datasets
  for use with GIGL.

### **Break** (15m)

### **Hands-on with GIGL**: Training and Inferring Industry-Scale GNNs - Part II (40m)

- **Training Logic Setup:** This part will focus on setting up the core logic for training and performing inference with
  GNNs using GIGL.
- **Configuration Setup:** We'll walk through configuring various parameters and settings within GIGL for optimal
  performance.

### **Break** (15m)

### **Hands-on with GIGL**: Training and Inferring Industry-Scale GNNs - Part III (40m)

- **End-to-End Pipeline Runs with Large-Scale Graph Data:** This section will involve running complete GNN pipelines on
  large datasets, from data loading to model evaluation.
- **Customization Potential:** We'll explore the possibilities for customizing GIGL to adapt it to specific research or
  application needs.


## Setup Resource Config.
The tutorial requires you to setup a [resource config](../../../docs/user_guide/config_guides/resource_config_guide.md) in order to launch jobs on GCP.

You may run the below from GiGL root to generate an appropriate resource config.
You may find the Project and User on the left hand panel on the qwiklabs page.


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

Accept the default region `us-central1` and output the resource config somewhere locally, like `examples/tutorial/KDD_2025/resource_config.yaml`.

## Resources

- **GiGL KDD '25 paper:** [GiGL: Large-Scale Graph Neural Networks at Snapchat](https://arxiv.org/abs/2502.15054)
- **[GiGL Documentation](../../../docs/user_guide/index.rst)**
