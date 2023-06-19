# mmmt examples

In [mmmt_examples](mmmt_examples/README.md) we keep a list of examples of MMMT applications.

The goal of these scripts is to showcase how to use MMMT to selected datasets.

## Structure in mmmt_examples
Each selected dataset has a folder in `mmmt_examples` containing
1. pipeline

    a runnable script going from data loading to model training and evaluation.

    *It requires installation of MMMT, see [MMMT readme](./../README.md)*

2. scripts specific to the selected dataset

    in particular, fuse operators for dataset loading and evaluation

In addition, to the KNIGHT example we also provide a **demonstration notebook**: a runnable notebook containing same functionality as the pipeline and adding visualizations of the generated sample graphs.


## Datasets
The code needed to download the selected datasets is not part of mmmt_example, but instructions are given in the pipeline and demonstration notebook.

### Datasets used so far in [mmmt_examples](mmmt_examples/README.md)
| Dataset name | Short description                 | Link to dataset                   |
|--------------|-----------------------------------|-----------------------------------|
| KNIGHT   | Kidney clinical Notes and Imaging to Guide and Help personalize Treatment and biomarkers discovery  | https://research.ibm.com/haifa/Workshops/KNIGHT/  |


## Pipeline configuration

Pipeline configuration is specified in the configuration yaml file.

In the current examples we configure:
1. How to process the cache (`cache`)
2. How to read the data (`data`)
3. How to encode each modality independently (`modality_encoding_strategy`)
4. How to fuse the encoded modalities (`fusion_strategy`)
5. How to solve the task (`task_strategy`)


### As a reference we list below configuration examples for the 6 graph modules currently available in MMMT for the solution of the task

The following arguments belong to `args.model_config.graph_model` of the object `MultimodalGraphModel`.

1. Multiplex GIN

```yaml
    module_identifier: "mplex"
    n_layers: *n_layers
    node_emb_dim: 1
```

2. Multiplex GCN

```yaml
    module_identifier: "mplex-prop"
    n_layers: *n_layers
    gl_hidden_size:
        - 2
        - 2
    node_emb_dim: 1
```

3. mGNN

```yaml
    module_identifier: "mgnn"
    n_layers: *n_layers
    gl_hidden_size:
        - 2
    num_att_heads: 4
    node_emb_dim: 1
```

4. MultiBehavioral GNN


```yaml
    module_identifier: "multibehav"
    n_layers: *n_layers
    gl_hidden_size:
        - 2
    node_emb_dim: 1
```



5. GCN


```yaml
    module_identifier: "gcn"
    n_layers: *n_layers
    gl_hidden_size:
        - 2
    node_emb_dim: 1
```


6. R-GCN

```yaml
    module_identifier: "rgcn"
    n_layers: *n_layers
    gl_hidden_size:
        - 2
    num_bases: 8
    node_emb_dim: 1
```
