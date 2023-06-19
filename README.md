# multimodal-model-toolkit


**multimodal-model-toolkit (MMMT, pronounced mammut)** is a platform for accelerating research and development with data in multiple
modalities, from data pre-processing, to model evaluation.

MMMT has a modular structure and distinguishes the following modules coordinated by one pipeline:
1. data loading
2. data representation
   1. unimodal representation
   2. multimodal representation
3. training
4. inference
5. evaluation

## Setup
<!-- See `python_requires` in setup.py/cfg -->
MMMT supports Python 3.8 and 3.9. To install:
```sh
pip install git+ssh://git@github.ibm.com/BiomedSciAI-Innersource/multimodal-model-toolkit
```

If you are contributing to the development of MMMT, see [dev_guide.md](dev_guide.md).

## Current methods for fusing representation
List of the methods currently integrated in the toolkit:

| Method | Short description | Link to publication                                |
|--------|-------------------|----------------------------------------------------|
| multiplex_gcn | multiplex GCN for message passing according to sGCN Conv for sparse graphs               | https://arxiv.org/abs/2210.14377         |
| multiplex_gin | multiplex GIN framework for message passing via multiplex walks               |   early https://arxiv.org/abs/2210.14377     |
| relational_gcn | relational GCN               | https://arxiv.org/pdf/1703.06103.pdf               |
| gcn    | baseline GCN               |   https://arxiv.org/abs/1609.02907v4  |
| mgnn    | mGNN framework for message passing               | https://arxiv.org/abs/2109.10119                   |
| multi_behavioral_gnn | multibehavioral GNN framework for message passing               | https://dl.acm.org/doi/pdf/10.1145/3340531.3412119 |


## User interface
In order to simplify the configuration of a computation using MMMT, we use the concept of pipeline, which can be fully specified using a `yaml` file.

The basic concept is that the yaml file describes the phases of the computations (e.g. data loading) and each phase contains the list of steps to be executed, specifying which object to use and the values of the arguments to give.

Beside clear modularization, this enables the user to launch multiple computations by just calling the same starting script (e.g. [full_mmmt_pipeline.py](mmmt_examples/knight/full_mmmt_pipeline.py)) passing different yaml files.

Default configuration values for a computation involving all possible phases are [here](mmmt/pipeline/defaults.yaml).


## Examples
In [mmmt_examples](mmmt_examples/README.md) we keep a list of examples of MMMT applications.

The goal of these scripts is to showcase how to use MMMT to selected datasets.

### Datasets used so far in [mmmt_examples](mmmt_examples/README.md)
| Dataset name | Short description                 | Link to dataset                   |
|--------------|-----------------------------------|-----------------------------------|
| KNIGHT   | Kidney clinical Notes and Imaging to Guide and Help personalize Treatment and biomarkers discovery  | https://research.ibm.com/haifa/Workshops/KNIGHT/  |
