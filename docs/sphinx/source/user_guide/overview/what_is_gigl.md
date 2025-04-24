# What is GiGL?

GiGL (Gigantic Graph Learning) is an open-source library designed for training and inference of Graph Neural Networks
(GNNs) at a very large scale, capable of handling billion-scale heterogeneous and feature-rich graph data. This library
provides:

- support of both supervised and unsupervised machine learning applications on graph data - including tasks like node
  classification, link prediction, self-supervised node representation learning, etc.
- abstracted designs to gracefully interface with modern ML libraries such as [PyTorch](https://pytorch.org/) and
  [TensorFlow](https://www.tensorflow.org/)
- flexible GNNs modeling with commonly used graph ML modeling libraries such as
  [PyG](https://github.com/pyg-team/pytorch_geometric) and [DGL](https://github.com/dmlc/dgl),
- utilities to enable data transformation, orchestration and pipelining of GNN workflows, which are particularly useful
  in large-scale deployments and recurrent applications.

At high level, GiGL abstracts the complicated and distributed processing of the gigantic graph data aways from the
users, such that users can focus on the graph ML modeling with the open-source libraries that they might already be
familiar with (PyG, DGL, etc). For more background, please check out our [research blog](<>) and [paper](<>).

# Why use GiGL?

GiGL was designed to address a single need: to enable ML researchers, engineers, and practitioners explore and iterate
on state-of-the-art graph ML models on large-scale graph data without having to stray far from the familiarities of
common open-source modeling libraries like PyG and DGL which are widely adopted in the research community. These
libraries have immense community support, especially for modeling advances and allowing native state-of-the-art GNN
research implementations. However, using these libraries to scale up to extremely large graphs beyond million-scale is
challenging and non-trivial.

GiGL is designed to interface cleanly with these libraries (for the benefit of ML researchers and engineers), while
handling scalability challenges with distributed data transformation, subgraph sampling, and persistence
behind-the-scenes with performant distributed libraries which the end-user needs minimal understanding of. Moreover,
GiGL enables large-scale component re-use, i.e. allowing for settings where expensive transformation and subgraph
sampling operations required to train GNNs can be generated once and re-used for hyperparameter tuning, experimentation,
and iteration multiple times with large cost-amortization potential. This can be hugely useful in maintaining low and
efficient cost-profiles in production settings at both training and inference-time, where multiple team-members are
iterating on models or multiple production inference pipelines are running.

If you are a ML practitioner, engineer, and/or enthusiast who is interested in working with and deploying GNNs on very
large-scale graph data, you will find GiGL useful.

# Why *not* use GiGL?

GiGL is designed with large-scale GNN settings in mind. For academic benchmarking and experimentation on smaller graphs
(ones that can be easily fit in RAM), open-source modeling libraries and built-in abstractions within PyG and DGL for
data transformation, subgraph sampling, data splitting, training and inference may be suitable and easier to use
directly. There is overhead introduced in GiGL compared to these libraries for distributed environment setup and
execution, and while this overhead is marginal in proportion to the benefits in large-scale scenarios, it may be
outsized in small-scale ones.

# How does GiGL work?

GiGL is designed with **horizontal-scaling** across many compute resources in mind, which makes it a good fit to run on
custom compute clusters and cloud-offerings. While vertically scaled GNN solutions are feasible to a limit, thanks to
advances in memory, core and GPU capacity in single machines, this limit is quickly saturated and needs frequent
revision as we consider graphs with more nodes, edges, and feature-rich data. Horizontal scaling is a resilient ideology
which in principle (and in practice, as we have found) allows for elastically scaling resources based on needs, which is
particularly appealing when considering deployment settings where scale of data can change rapidly with e.g. user or
platform growth.

GiGL is comprised of the following 5 components ([more details on them](.components.md)):

- **Data Preprocessor** Distributed feature transformation pipeline which allows for feature-scaling, normalization,
  categorical-feature handling (encoding, vocabulary inference), and more.
- **Subgraph Sampler** Distributed subgraph generation pipeline which enables custom graph sampling specifications to
  dictate message-passing flow, and custom sample generation (e.g. handling positive/negative sampling) to facilitate
  tasks like node classification and link prediction.
- **Split Generator** Distributed data splitting routine to generate globally consistent train, validation and test
  splits according to flexible split strategies (transductive, inductive, custom, time-based, etc.)
- **Trainer** Distributed model trainer which consumes data output by Split Generator to do model training, validation
  and testing dictated by user code to generate model artifacts.
- **Inferencer** Distributed model inference which generates embeddings and/or class predictions dictated by user code.

Each of these components is run in specification with some combination of user code for flexibility, and a configuration
file which is partially user-specified and partially auto-generated with a precursor component called Config Populator
which houses the details on what logic will run, and where the resulting intermediate and final assets (transformed
data, splits, model artifact, inferences) will be stored.
