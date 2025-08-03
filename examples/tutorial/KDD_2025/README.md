# <p align="center"> Training Industry-scale GNNs with GiGL: KDD'25 Hands-On Tutorial </p>

## <p align="center">Sunday, August 3rd, 1 - 4pm EDT, Room 707 </p>

______________________________________________________________________

**1 billion nodes** connected by over **100 billion edges**, and petabytes of daily ingested data - that’s the kind of
complexity and scale we’re up against when training GNNs at industry-scale.

In this tutorial, we’ll showcase you how industry-scale GraphML can be achieved with GiGL.

## Table of Contents

- [Tutorial Goal](#tutorial-goal)
- [Tutorial Schedule](#tutorial-schedule)
  - [GNNs and their Scale Challenges](#gnns-and-their-scale-challenges)
  - [Overview of GIGL](#overview-of-gigl)
  - [Hands-on with GIGL: Training and Inferring Industry-Scale GNNs](#hands-on-with-gigl-training-and-inferring-industry-scale-gnns)
- [Hands-On Lab Instructions](#hands-on-lab-instructions)
- [Resources](#resources)
- [In-Person Presenters](#in-person-presenters)

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

| Time (EDT) | Topic                                  | Presenter        | Materials                                   |
| ---------- | -------------------------------------- | ---------------- |---------------------------------------------|
| 1:00 PM    | Introduction                           | Neil Shah        | [Slides](slides_tutorial_KDD_25.pdf)        |
| 1:10 PM    | GNNs and their Scale Challenges        | Neil Shah        | [Slides](slides_tutorial_KDD_25.pdf)        |
| 1:25 PM    | Overview of GiGL                       | Yozen Liu        | [Slides](slides_tutorial_KDD_25.pdf)        |
| 1:40 PM    | Hands-on with GiGL - Show case & Setup | Yozen Liu        | -                                           |
| 1:50 PM    | Hands-on with GiGL - Tabularization    | Shubham Vij      | [Notebook]()                                |
| 2:20 PM    | Hands-on with GiGL - In-memory         | Kyle Montemayor  | [Notebook](heterogeneous_walktrhough.ipynb) |
| 3:00 PM    | Coffee break (30m)                     | -                | -                                           |
| 3:30 PM    | Hands-on with GiGL - Customization     | Matthew Kolodner | [Slides](slides_tutorial_KDD_25.pdf)        |
| 3:55 PM    | Conclusion                             | Yozen Liu        | [Slides](slides_tutorial_KDD_25.pdf)        |

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

## Hands-On Lab Instructions

To access our lab environment, scan the QR code displayed on the presentation screen, type in your email, then enter the
OTP code displayed.

After registration, click the link displayed to access the lab (you might have to login with your registered email). Now
you should be able to access the labs!

In our lab environment, you will have access to your own GCP project to run our hands-on tutorial.

<img width="2560" height="1440" alt="Screenshot 2025-07-31 at 11 14 40 AM" src="https://github.com/user-attachments/assets/33190067-2410-4f8f-adba-15afd328ee99" />
<img width="2560" height="1440" alt="Screenshot 2025-07-31 at 11 14 49 AM" src="https://github.com/user-attachments/assets/4e46dcf2-29c1-447b-8faf-16b94c0d613e" />

Follow the steps in our [lab instructions](lab_instructions.md) to set up your hands-on lab in the Qwiklabs environment.
If you have any questions, please raise your hand, our tutors will be able to help.

## Resources

- **Hands-On Tutorial instructions:** [GiGL KDD '25 Hands-On Tutorial](lab_instructions.md)
- **GiGL KDD '25 paper:** [GiGL: Large-Scale Graph Neural Networks at Snapchat](https://arxiv.org/abs/2502.15054)
- **GiGL Documentation:**
  - [GiGL User Guide](https://snapchat.github.io/GiGL/docs/user_guide/index.html)
  - [GiGL API Reference](https://snapchat.github.io/GiGL/docs/api/index.html)

# In-Person Presenters

### [Neil Shah](https://nshah.net/)

Dr. Neil Shah is a Principal Scientist at Snap Research. His research focuses on graph ML, large-scale representation
learning, and recommender systems. His work has resulted in 70+ refereed publications at top data mining and machine
learning venues.

He has also served as an organizer across multiple venues including KDD, WSDM, SDM, ICWSM, ASONAM and more, and received
multiple best paper awards (KDD, CHI), departmental rising star awards (NCSU), and outstanding service and reviewer
awards (NeurIPS, WSDM). He has also served as an organizer across multiple workshops and tutorials at KDD, AAAI, ICDM,
CIKM and more.

### [Yozen Liu](https://scholar.google.com/citations?user=i3U2JjEAAAAJ&hl=en)

Yozen Liu is a Senior Research Engineer at Snap Research, focusing on graph machine learning, user modeling,
recommendation systems and their industrial applications. He holds an MS in Computer Science from the University of
Southern California and has published 25+ papers at top-tier data mining, machine learning and information retrieval
conferences such as KDD, ICLR, NeurIPS, ICML, WWW, and SIGIR.

### Shubham Vij

Shubham is a Staff Research Engineer specializing in engineering ML systems - bridging the gap between theoretical
advancements and enterprise scale product development challenges in GraphML, NLP, CV, and ML OPS. He holds dual
Bachelors degrees in Computer Science, and Business Administration from University of Waterloo, and Wilfrid Laurier
University respectively.

### Kyle Montemayor

Kyle Montemayor is a Research Engineer at Snap Research, focusing on scaling graph machine learning techniques and
productionizing graph machine learning research. Prior to joining Snap Research, he worked on Tensorflow Extended (TFX)
at Google. He holds a B.S. of Computer Engineering from the University of Maryland, College Park.

### Matthew Kolodner

Matthew is a Research Engineer at Snap Research. Matthew's work at Snap focuses on graph machine learning, particularly
in large-scale applications. Matthew received his B.S. and M.S. in Computer Science from Stanford University with a
specialization in Artificial Intelligence.
