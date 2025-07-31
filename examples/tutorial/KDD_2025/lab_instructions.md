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

1. Click **Start Lab** on the left side of the page. Wait for the lab to be set up.
2. Right click on **Open Google Cloud Console** button and open in a incognito window.
3. Agree to the terms of service for your custom student account for the duration of the lab.
4. Use the top search bar or left side bar to navigate to the **Workbench** page provided by **Vertex AI**.

<img width="1260" height="777" alt="Screenshot 2025-07-31 at 10 39 58 AM" src="https://github.com/user-attachments/assets/21656547-5889-4c00-9127-ccb8cc4deaf7" />

5. There should be an existing Workbench instance, click open **Jupyterlab**.

<img width="997" height="458" alt="Screenshot 2025-07-31 at 10 48 37 AM" src="https://github.com/user-attachments/assets/3f31655a-d4c1-4545-9616-3285e9892f31" />

6. Open the `gigl` folder in the left sidebar. Navigate to the `examples/tutorial/KDD_2025` folder.
7. In the hands-on portion of the tutorial, we will be running notebooks
   `examples/toy_visual_example/toy_example_walkthrough.ipynb` for tabularization subgraph sampling, and
   `examples/tutorial/KDD_2025/heterogeneous_walkthrough.ipynb` for in-memory subgraph sampling.
8. Open the notebooks with `gigl` kernel. The select kernel will show on the top right corner of the notebook page.

<img width="1079" height="623" alt="Screenshot 2025-07-31 at 10 53 57 AM" src="https://github.com/user-attachments/assets/c7046ec1-ed79-445b-873e-f15eff7f9d2f" />

9. Pro tip: To enable scrolling for notebook cells, you can ctrl/cmd + A to select all notebook cells, right-click then
   select `Enable Scrolling For Outputs`.
10. Follow along the tutorial presentation and learn about how to use GiGL! 

## Additional Resources

- GiGL KDD '25 tutorial page:
  [Homepage](https://github.com/Snapchat/GiGL/blob/main/examples/tutorial/KDD_2025/README.md)
- GiGL KDD ADS track paper: [Paper link](https://arxiv.org/abs/2502.15054)
- GiGL library source code: [GitHub](https://github.com/Snapchat/GiGL/tree/main)
- GiGL documentation: [Read the Docs](https://snapchat.github.io/GiGL/index.html)
