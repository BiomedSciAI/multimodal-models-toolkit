import os
import copy
from fuse.utils.file_io.file_io import save_dataframe

from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.utils.file_io.file_io import create_dir
import dgl
import mmmt
import torch
import mlflow
from fuse.utils.ndict import NDict


def custom_collate_graph(graph_batch):
    """
    Custom collate to use for batching of the derived graphs

    Args:
        graph_batch: a list of graphs to be batched

    Returns:
        dgl batched graphs
    """
    tmp1 = []
    tmp2 = []
    for derived_graph in graph_batch:
        tmp1.append(derived_graph[0])
        if len(derived_graph) > 1:
            tmp2.append(derived_graph[1])
    if tmp2:
        return (dgl.batch(tmp1), dgl.batch(tmp2))
    else:
        return (dgl.batch(tmp1),)


def custom_collate_tensor(f_batch):
    """
    Custom collate to use for batching of the node features of the derived graphs

    Args:
        f_batch: a list of node features to be batched

    Returns:
        batched node features
    """
    node_features = torch.tensor([])
    for bf in f_batch:
        node_features = torch.cat((node_features, torch.tensor(bf)), 0)
    # node_features = torch.tensor(node_features)

    return node_features


class MultimodalGraphModel:
    """
    Construction, training and inference of the multimodal graph model.
    """

    def __init__(self, args_dict):
        """
        Args:
            graph_train_dataset: training dataset for the multimodal graph model
            graph_validation_dataset: validation dataset for the multimodal graph model
            graph_model_configuration: user input, configuration of multimodal graph model
        """

        self.train_config = copy.deepcopy(args_dict["step_args"]["training"])
        self.step_args = args_dict["step_args"]

        graph_train_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["fused_dataset_key"]
        ]["graph_train_dataset"]
        graph_validation_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["fused_dataset_key"]
        ]["graph_validation_dataset"]
        self.graph_test_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["fused_dataset_key"]
        ]["graph_test_dataset"]

        self.root_dir = args_dict["root_dir"]

        graph_module = args_dict["step_args"]["model_config"]["graph_model"]

        self.graph_key = args_dict["step_args"]["io"]["input_key"]
        self.target_key = args_dict["step_args"]["io"]["target_key"]
        self.head_hidden_size = args_dict["step_args"]["model_config"]["head_model"][
            "head_hidden_size"
        ]
        self.dropout = args_dict["step_args"]["model_config"]["head_model"]["dropout"]
        self.add_softmax = args_dict["step_args"]["model_config"]["head_model"].get(
            "add_softmax", True
        )

        self.batch_size = args_dict["step_args"]["training"]["batch_size"]
        del self.train_config["batch_size"]

        self.obj_reg = args_dict["object_registry"]
        self.train_config["train_metrics"] = {
            args_dict["step_args"]["training"]["train_metrics"][
                "key"
            ]: self.obj_reg.instance_object(
                args_dict["step_args"]["training"]["train_metrics"]["object"],
                args_dict["step_args"]["training"]["train_metrics"]["args"],
            )
        }
        self.train_config["validation_metrics"] = {
            args_dict["step_args"]["training"]["validation_metrics"][
                "key"
            ]: self.obj_reg.instance_object(
                args_dict["step_args"]["training"]["validation_metrics"]["object"],
                args_dict["step_args"]["training"]["validation_metrics"]["args"],
            )
        }
        self.model_dir = os.path.join(self.root_dir, self.train_config["model_dir"])
        self.train_config["model_dir"] = self.model_dir

        self.num_workers = args_dict["num_workers"]
        self.out_size = args_dict["step_args"]["model_config"]["num_classes"]

        self.model_out_key = args_dict["step_args"]["io"]["prediction_key"]

        self.test_results_filename = args_dict["step_args"]["testing"][
            "test_results_filename"
        ]
        self.evaluation_directory = args_dict["step_args"]["testing"][
            "evaluation_directory"
        ]

        self.checkpoint_filename = "best_epoch.ckpt"

        self.mb_trainer = None

        # Configure Graph Module
        graph_sample = graph_train_dataset[0][self.graph_key]["graph"]

        MC = mmmt.models.graph.module_configurator.ModuleConfigurator(graph_module)
        graph_model, head_in_size, head_num_nodes = MC.get_module(graph_sample)
        head_model = mmmt.models.head.mlp.MLP(
            head_num_nodes,
            head_in_size,
            self.head_hidden_size,
            self.out_size,
            self.dropout,
            self.add_softmax,
        )
        self.build_model(graph_model, head_model)

        # Configure data loaders
        graph_train_dataloader = DataLoader(
            dataset=graph_train_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(
                special_handlers_keys={
                    self.graph_key + ".graph": custom_collate_graph,
                    self.graph_key + ".node_features": custom_collate_tensor,
                }
            ),
            num_workers=self.num_workers,
        )

        graph_validation_dataloader = DataLoader(
            dataset=graph_validation_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(
                special_handlers_keys={
                    self.graph_key + ".graph": custom_collate_graph,
                    self.graph_key + ".node_features": custom_collate_tensor,
                }
            ),
            num_workers=self.num_workers,
        )

        self.graph_train_dataloader = graph_train_dataloader
        self.graph_validation_dataloader = graph_validation_dataloader

    def build_model(self, graph_model, head_model):
        """
        Multimodal graph model construction from the graph model and the head model.

        Args:
            graph_model: graph model, where message passing and aggregation is happening
            head_model: head model, where signal is aggragated up to a target
        """

        MB = mmmt.models.model_builder.ModelBuilder(
            graph_model, head_model, self.batch_size
        )

        self.mb_trainer = mmmt.data.representation.ModelBuilderTrainer(
            MB,
            self.graph_key,
            self.model_out_key,
            self.target_key,
        )

    def __call__(self):
        with mlflow.start_run(run_name=f"{self.__class__.__qualname__}", nested=True):
            mlflow.log_params(NDict(self.step_args).flatten())
            self.train()
            self.test()

    def train(
        self,
    ):
        """
        Multimodal graph model training

        Returns:
            the model with the weights from the best epoch, measured using the validation set
        """

        self.mb_trainer.set_train_config(**self.train_config)

        self.mb_trainer.fit(
            self.graph_train_dataloader, self.graph_validation_dataloader
        )

    def test(
        self,
    ):
        """
        Apply model and extract both output and ground-truth labels

        Args:
            graph_infer_dataset: inference dataset for the multimodal graph model
            output_filename: file name where the inference results are stored
            checkpoint_filename: file name where the model is cached
        """

        # run inference and eval
        eval_dir = os.path.join(self.root_dir, self.evaluation_directory)
        create_dir(eval_dir)

        output_filename = os.path.join(eval_dir, self.test_results_filename)

        graph_infer_dataloader = DataLoader(
            dataset=self.graph_test_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(
                special_handlers_keys={
                    self.graph_key + ".graph": custom_collate_graph,
                    self.graph_key + ".node_features": custom_collate_tensor,
                }
            ),
            num_workers=self.num_workers,
        )
        test_df = self.mb_trainer.predict(
            graph_infer_dataloader,
            self.model_dir,
            os.path.join(self.model_dir, self.checkpoint_filename),
            [self.target_key, self.model_out_key],
        )
        test_df = test_df.rename(
            columns={self.target_key: "target", self.model_out_key: "pred"}
        )
        save_dataframe(test_df, output_filename)
        mlflow.log_artifact(output_filename)

        return test_df
