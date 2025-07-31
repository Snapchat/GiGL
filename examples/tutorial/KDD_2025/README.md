# <p align="center"> Training Industry-scale GNNs with GiGL: KDD'25 Hands-On Tutorial </p>

## <p align="center">Sunday, August 3rd, 1 - 4pm EDT, Room 707 </p>

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

## Tutorial Schedule

| Time (EDT) | Topic                                  | Presenter        | Materials |
| ---------- | -------------------------------------- | ---------------- | --------- |
| 1:00 PM    | Introduction                           | Neil Shah        | Slides    |
| 1:10 PM    | GNNs and their Scale Challenges        | Neil Shah        | Slides    |
| 1:25 PM    | Overview of GiGL                       | Yozen Liu        | Slides    |
| 1:40 PM    | Hands-on with GiGL - Show case & Setup | Yozen Liu        | -         |
| 1:50 PM    | Hands-on with GiGL - Tabularization    | Shubham Vij      | Notebook  |
| 2:20 PM    | Hands-on with GiGL - In-memory         | Kyle Montemayor  | Notebook  |
| 3:00 PM    | Coffee break (30m)                     | -                | -         |
| 3:30 PM    | Hands-on with GiGL - Customization     | Matthew Kolodner | Slides    |
| 3:55 PM    | Conclusion                             | Yozen Liu        | Slides    |

### GNNs and their Scale Challenges

- **GNN Fundamentals and Formalisms:** This section will cover the basics of Graph Neural Networks, including their
  underlying mathematical concepts and how they operate.
- **Real-world Applications:** We'll explore various practical applications of GNNs across different industries,
  showcasing their versatility and impact.
- **Scalability Challenges:** This part will delve into the inherent difficulties of scaling GNNs to handle large,
  real-world datasets, highlighting the computational and memory limitations.

### Overview of GIGL

- **Technical Scaling Strategy:** We'll discuss the technical approaches and strategies employed by GIGL to overcome the
  scalability challenges of GNNs.
- **Library Design:** This section will provide an overview of GIGL's architectural design, explaining how its
  components are structured and interact.
- **GIGL Core Components:** We'll identify and describe the key building blocks and functionalities that make up the
  GIGL library.

### **Hands-on with GIGL**: Training and Inferring Industry-Scale GNNs

- **Environment Setup:** This hands-on segment will guide you through setting up the necessary software and hardware
  environment to work with GIGL.
- **Provisioning Access to Large-Scale Graph Data:** We'll cover how to access and prepare large-scale graph datasets
  for use with GIGL.
- **Training Logic Setup:** This part will focus on setting up the core logic for training and performing inference with
  GNNs using GIGL. We will also walk through two different paradigms of subgraph sampling, e.g. **Tabularization** &
  **In-memory**.
- **Configuration Setup:** We'll walk through configuring various parameters and settings within GIGL for optimal
  performance.
- **End-to-End Pipeline Runs with Large-Scale Graph Data:** This section will involve running complete GNN pipelines on
  large datasets, from data loading to model evaluation.
- **Customization Potential:** We'll explore the possibilities for customizing GIGL to adapt it to specific research or
  application needs.

### Hands-On Lab Instructions

To access our lab environment, scan the QR code displayed on the presentation screen, type in your email, then enter the
OTP code displayed.

After registration, click the link displayed to access the lab (you might have to login again with your registered email). Now you should be able to access the labs!

In our lab environment, you will have access to your own GCP project to run our hands-on tutorial.

<img width="2560" height="1440" alt="Screenshot 2025-07-31 at 11 14 40 AM" src="https://github.com/user-attachments/assets/33190067-2410-4f8f-adba-15afd328ee99" />
<img width="2560" height="1440" alt="Screenshot 2025-07-31 at 11 14 49 AM" src="https://github.com/user-attachments/assets/4e46dcf2-29c1-447b-8faf-16b94c0d613e" />

Follow the steps in our [lab instructions](lab_instructions.md) to set up your hands-on lab in the Qwiklabs environment.
If you have any questions, please raise your hand, our tutors will be able to help.

## Resources

- **Hands-On Tutorial instructions:** [GiGL KDD '25 Hands-On Tutorial](lab_instructions.md)
- **GiGL KDD '25 paper:** [GiGL: Large-Scale Graph Neural Networks at Snapchat](https://arxiv.org/abs/2502.15054)
- **GiGL Documentation:**
  - [GiGL User Guide](https://snapchat.github.io/GiGL/docs/user_guide/index.html)
  - [GiGL API Reference](https://snapchat.github.io/GiGL/docs/api/index.html)
