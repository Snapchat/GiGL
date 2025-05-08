# GiGL Components

GiGL contains six components, each designed to facilitate the platforms end-to-end graph machine learning (ML) tasks.
The components are as follows:

| Component         | Source Code                                                               | Documentation                                           |
| ----------------- | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| Config Populator  | {py:class}`gigl.src.config_populator.config_populator.ConfigPopulator`    | [here](docs/user_guide/components/config_populator.md)  |
| Data Preprocessor | {py:class}`gigl.src.data_preprocessor.data_preprocessor.DataPreprocessor` | [here](docs/user_guide/components/data_preprocessor.md) |
| Subgraph Sampler  | {py:class}`gigl.src.subgraph_sampler.subgraph_sampler.SubgraphSampler`    | [here](docs/user_guide/components/subgraph_sampler.md)  |
| Split Generator   | {py:class}`gigl.src.split_generator.split_generator.SplitGenerator`       | [here](docs/user_guide/components/split_generator.md)   |
| Trainer           | {py:class}`gigl.src.training.trainer.Trainer`                             | [here](docs/user_guide/components/trainer.md)           |
| Inferencer        | {py:class}`gigl.src.inference.inferencer.Inferencer`                      | [here](docs/user_guide/components/inferencer.md)        |

The figure below illustrates at a high level how all the components work together.
(<span style="color:purple">Purple</span> items are work-in-progress.)

<img src="docs/assets/images/gigl_system_fig.png" alt="GiGL System Figure" width="50%" />

The figure below is a example GiGL workflow with tabularized subgraph sampling for the task of link prediction, in which
the model is trained with triplet-style contrastive loss on a set of anchor nodes along with their positives and
(in-batch) negatives.

![gigl_nablp](docs/assets/images/gigl_nablp.png)

## Config Populator

<img src="../../assets/images/config_populator_icon.png" height="100px">

Component for processing template config files and updating with fields that are needed for downstream components.
[More Details](../components/config_populator)

______________________________________________________________________

## Data Preprocessor

<img src="../../assets/images/data_preprocessor_icon.png" width="100px">

Component for reading and processing node, edge, and feature data and transforming it as needed for downstream
components. [More Details](../components/data_preprocessor)

______________________________________________________________________

## Subgraph Sampler

<img src="../../assets/images/subgraph_sampler_icon.png" height="100px">

Component that generates k-hop localized subgraphs for each node in the graph.
[More Details](../components/subgraph_sampler)

______________________________________________________________________

## Split Generator

<img src="../../assets/images/split_generator_icon.png" width="100px">

Component to split the data into training, validation, and test sets. [More Details](../components/split_generator)

______________________________________________________________________

## Trainer

<img src="../../assets/images/trainer_icon.png" width="100px">

Component to run distributed training either locally or on the cloud. [More Details](../components/trainer)

______________________________________________________________________

## Inferencer

<img src="../../assets/images/inferencer_icon.png" width="100px">

Component that runs inference to generate output embeddings and/or predictions [More Details](../components/inferencer)
