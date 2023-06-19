# User guide

This guide will take you through the process for training a multimodal model for your dataset.
You can do that using `mmmt` as a normal python library, make sure to check the API for that.
But if you want a simpler way to just create a multimodal representation and a model with minimal code writing,
you can leverage the pipeline concept.


An `MMMTPipeline` is a workflow that aims at creating a model with minimal code writing and a `yaml` configuration file.
The reason for using a such a file is that hyperparameter tuning is done without rewriting code, you just need to modify a text file.
The pipeline object will perform a number of steps of the following categories:
  - [data](#data)
  - modality_encoding_strategy
  - fusion_strategy
  - task strategy

Additionally it contains two more items that are used for configuring the environment:
  - cache
  - mlflow


## Minimal pipeline code
A python file containing the following lines is enough to successfully run a pipeline

```
MMMTP = MMMTPipeline("multimodal_example.yaml", specific_objects)
MMMTP.run_pipeline()

```

## Adding specific objects to the object registry

The pipeline object leverages an `ObjectRegistry` that contains all the default modules from the core library.
The user can extend the objects that are callable by the pipeline through a dictionary of `specific_objects`.
It provides the pipeline access to user-defined callable objects that are needed for a specific (hence the name) task. e.g., a specific object can be a callable object that preprocesses the data. For instance:

```python
specific_objects = {
        "static_pipeline": {
            "object": TCGACSVDataset.static_pipeline,
        },
        "get_splits": {
            "object": TCGACSVDataset.get_splits,
        },
    }
```


## data

`mmmt` supports a large variety of formats and datasets, because it builds on [FuseMedML](https://github.com/BiomedSciAI/fuse-med-ml) for the dataset preparation pipeline.
As shown in the next code block, we start with a `fuse_object` called `static pipeline`, and then a non-fuse `object` called `get_splits`. These names are the keys used in the `specific_objects` dictionary passed to the pipeline.
```yaml
data:
  - fuse_object: "static_pipeline"
    args:
      name: "Test dataset"
  - object: "get_splits"
    args:
      name: "splits"
    io:
      input_key: null
      output_key: "data_splits"
```

### diference between fuse_objects and objects
FUSE objects are objects that inherit from the `PipelineDefault` FuseMedML class.
This allows some interesting caching features.

## modality_encoding_strategy
The modality encoding strategy serves two purposes:
- Represent each sample of a modality with a feature vector
- Reduce the dimensionality of this unimodal representation to reduce computational cost.

The following block shows an example of modality encoding strategy. In this example we use the standard MMMT object `ModalityEncoding` which allows obtain unimodal representations of the each modality. It allows to use a pretrained model (a `torch.nn.Module`), train an autoencoder setting `use_autoencoder: True`, both or none of them (e.g. for data that has been already preprocessed separately and is the the correct representation)
```yaml
modality_encoding_strategy:
  - object: "ModalityEncoding"
    args:
      data.input.raw.modality1:
        model: null
        output_key: "data.input.encoded.modality1"
        use_autoencoder: True
        encoding_layers:
          - 128
          - 64
        use_pretrained: False
        batch_size: 5
        training:
          model_dir: "model_modality1"
          pl_trainer_num_epochs: 1
          pl_trainer_accelerator: "cpu"
      data.input.raw.modality2:
        model_path: "path-to-model"
        add_feature_names: False
        dimensions: 2
        output_key: "data.input.encoded.modality2"
        use_autoencoder: False
      data.input.raw.modality3:
        ...
```

## fusion_strategy

The fusion strategy is the block that will combine the several unimodal representations into a common representation for all modalities.
This can be done through concatenation (early fusion) or through more sophisticated methods, like the Graph Rrepresentation method. The structure of the strategy contains a series of steps similar to the following example (details are given in the subsections):
```yaml
fusion_strategy:
  - object: "EncodedUnimodalToConcept"
    ...
  - object: "ConceptToGraph"
    ...
  - object: "GraphVisualization"
    ...
```
### Unimodal Representation to Concept
```yaml
fusion_strategy:
  - object: "EncodedUnimodalToConcept"
    args:
      use_autoencoders: True
      add_feature_names: False
      encoding_layers:
        - 32
        - &n_layers 16
      use_pretrained: False
      batch_size: 5
      training:
        model_dir: "model_concept"
        pl_trainer_num_epochs: 1
        pl_trainer_accelerator: "cpu"
      io:
        concept_encoder_model_key: "concept_encoder_model"
        input_keys:
          - "data.input.encoded.modality1"
          - "data.input.encoded.modality2"
          - "data.input.encoded.modality3"
        output_key: "data.input.concatenated"
```
In this example the `EncodeUnimodalToConcept` object will train an autoencoder that embeds the concatenated representation of the modalities indicated in `input_keys`. The `output key`of this object will store the concatenated representation of the features and but the model is stored in `concept_encoder_model_key`.

### Concept To Graph
```yaml
fusion_strategy:
  ...
  - object: "ConceptToGraph"
    args:
      module_identifier: &graph_module "mplex"
      thresh_q: 0.95
      io:
        concept_encoder_model_key: "concept_encoder_model"
        fused_dataset_key: "fused_dataset"
        input_key: "data.input.concatenated"
        output_key: "data.derived_graph"
  ...
```
These two keys are used by the `ConceptToGraph` object to build a graph on the concatenated unimodal features by analyzing their contribution to the concept space dimensions.
The `ConceptToGraph` object can produce various types of graphs, for which a specific model can be trained to solve the multimodal task. The following graph types are supported:

1. Multiplex GIN: `module_identifier: "mplex"`
1. Multiplex GCN: `module_identifier: "mplex-prop"`
1. mGNN: `module_identifier: "mgnn"`
1. MultiBehavioral GNN: `module_identifier: "multibehav"`
1. GCN: `module_identifier: "gcn"`
1. R-GCN: `module_identifier: "rgcn"`

A new dataset is created under the `fuse_dataset_key` that contains the graph representation in the `output_key`

### Graph Visualization
```yaml
fusion_strategy:
  ...
  - object: "GraphVisualization"
    args:
      selected_samples:
        graph_train_dataset:
          - 0
          - 1
        graph_validation_dataset: "all"
        graph_test_dataset: "all"
      feature_group_sizes:
        modality1: 64
        modality2: 32
        modality3: 64
      io:
        file_prefix: "graph_visualization"
        fused_dataset_key: "fused_dataset"
```
The user can generate a graph visualization for some or all of the samples of any of the dataset splits. E.g.: `graph_validation_dataset: "all"` will generate a visualization of the all the validation samples combined, whereas for the train set only the first two samples (`0,1`) will be combined.

The visualization requires that the user provides the number of features from each modality to group them by color

The visualizations are saved under the specified paths.

## task_strategy
Finally, the pipeline can perform a machine learning task on the multimodal representation. This is defined by the task strategy step.
The user may use a `MultimodalGraphModel` object or an `MultimodalMLP` object to perform this task.
Both of them require a set of common training and testing arguments, where the user can select how to choose the best epoch and what metric to compute in addition to the loss for each train and validation step.
The task module will perform a final prediction round on the test dataset and the results will be stored in the `evaluation_directory` under `test_results_filename`
### training and testing arguments
```yaml
task_strategy:
  - object: ...
    args:
      io:
        ...
        target_key: &target "data.ground_truth"
        prediction_key: &prediction "model.out"
      model_config:
        ...
      training:
        model_dir: "task_model"
        batch_size: 1
        best_epoch_source:
          mode: "max"
          monitor: "validation.metrics.accuracy"
        train_metrics:
          key: "accuracy"
          object: "MetricAccuracy"
          args:
            pred: *prediction
            target: *target
        validation_metrics:
          key: "accuracy"
          object: "MetricAccuracy"
          args:
            pred: *prediction
            target: *target
        pl_trainer_num_epochs: 500
        pl_trainer_accelerator: "cpu"
        pl_trainer_devices: 1

      testing:
        test_results_filename: &test_results_filename "test_results.pickle"
        evaluation_directory: &evaluation_directory "eval"

```


### MultimodalMLP
```yaml
- object: "MultimodalMLP"
    args:
      io:
        input_key: "data.input.concatenated"
        target_key: &target "data.ground_truth"
        prediction_key: &prediction "model.out"
      model_config:
        hidden_size:
          - 100
          - 20
        dropout: 0.5
        add_softmax: True
        num_classes: 2
      training:
        ...
      testing:
        ...

```
### MultimodalGraphModel

```yaml
task_strategy:
  - object: "MultimodalGraphModel"
    args:
      io:
        fused_dataset_key: "fused_dataset"
        input_key: "data.derived_graph"
        target_key: &target "data.ground_truth"
        prediction_key: &prediction "model.out"
      model_config:
        graph_model:
          module_identifier: *graph_module
          n_layers: *n_layers
          node_emb_dim: 1
        head_model:
          head_hidden_size:
            - 100
            - 20
          dropout: 0.5
          add_softmax: True
        num_classes: 11

      training:
        ...
      testing:
        ...
```

#### Supported Graph Neural Networks
`MultimodalGraphModel` supports six different flavors of graph neural networks for the ML task, they can be configured as follows:

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


## cache
The use of a cache simplifies running several experiments on the same data. The configuration for the cache is done as follows:
```yaml
cache:
  num_workers: 1
  restart_cache: True
  root_dir: "_examples/cache"
```


## mlflow
When starting an `MMMTPipeline` a new mlflow run will be created, using the MLFLOW environment variables or the configuration provided in this section of the yaml.
All of the steps are logged independently and the metrics and losses are tracked for further inspection.
Additionally, the models, the visualization and the test-time prediction files are also logged as artifacts.
```yaml
mlflow:
  MLFLOW_TRACKING_URI: null
  MLFLOW_EXPERIMENT_NAME: null
```
