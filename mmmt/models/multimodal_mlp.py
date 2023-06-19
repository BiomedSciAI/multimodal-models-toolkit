import os
import copy
from fuse.utils.file_io.file_io import save_dataframe

from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.utils.file_io.file_io import create_dir
import mmmt


class MultimodalMLP:
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

        train_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["input_key"]
        ]["concatenated_training_dataset"]
        validation_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["input_key"]
        ]["concatenated_validation_dataset"]
        self.test_dataset = args_dict["pipeline"][
            args_dict["step_args"]["io"]["input_key"]
        ]["concatenated_test_dataset"]

        self.root_dir = args_dict["root_dir"]

        self.model_in_key = args_dict["step_args"]["io"]["input_key"]
        self.target_key = args_dict["step_args"]["io"]["target_key"]
        self.model_out_key = args_dict["step_args"]["io"]["prediction_key"]

        self.hidden_size = args_dict["step_args"]["model_config"]["hidden_size"]
        self.dropout = args_dict["step_args"]["model_config"]["dropout"]
        self.add_softmax = args_dict["step_args"]["model_config"].get(
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

        self.test_results_filename = args_dict["step_args"]["testing"][
            "test_results_filename"
        ]
        self.evaluation_directory = args_dict["step_args"]["testing"][
            "evaluation_directory"
        ]

        self.checkpoint_filename = "best_epoch.ckpt"

        self.mb_trainer = None

        # Configure MLP Module
        num_feat = train_dataset[0][args_dict["step_args"]["io"]["input_key"]].shape[0]

        mlp_model = mmmt.models.head.mlp.MLP(
            num_feat,
            1,
            self.hidden_size,
            self.out_size,
            self.dropout,
            self.add_softmax,
        )

        self.build_model(mlp_model)

        # Configure data loaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )

        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def build_model(self, head_model):
        """
        Multimodal MLP model construction from the MLP model.

        Args:
            head_model: head model, where signal is aggragated up to a target
        """

        MB = head_model

        self.mb_trainer = mmmt.data.representation.ModelBuilderTrainer(
            MB,
            self.model_in_key,
            self.model_out_key,
            self.target_key,
        )

    def __call__(self):
        self.train()
        self.test()

    def train(
        self,
    ):
        """
        Multimodal MLP model training

        Returns:
            the model with the weights from the best epoch, measured using the validation set
        """

        self.mb_trainer.set_train_config(**self.train_config)

        self.mb_trainer.fit(self.train_dataloader, self.validation_dataloader)

    def test(
        self,
    ):
        """
        Apply model and extract both output and ground-truth labels

        """

        # run inference and eval
        eval_dir = os.path.join(self.root_dir, self.evaluation_directory)
        create_dir(eval_dir)

        output_filename = os.path.join(eval_dir, self.test_results_filename)

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            num_workers=self.num_workers,
        )
        test_df = self.mb_trainer.predict(
            test_dataloader,
            self.model_dir,
            os.path.join(self.model_dir, self.checkpoint_filename),
            [self.target_key, self.model_out_key],
        )
        test_df = test_df.rename(
            columns={self.target_key: "target", self.model_out_key: "pred"}
        )
        save_dataframe(test_df, output_filename)

        return test_df
