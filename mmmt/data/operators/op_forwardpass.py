from fuse.data.ops.op_base import OpBase
from typing import List, Union
from fuse.utils.ndict import NDict
import torch
import logging
import numpy as np


class OpForwardPass(OpBase):
    """
    Operator for applying a forward pass of a pretrained model.
    """

    def __init__(self, model, modality_dimensions: int, add_feature_names: bool = True):
        """Constructor method

        :param model: pretrained model to be applied
        :type model: torch.nn.Module
        :param modality_dimensions: dimensions required by the model
        :type modality_dimensions: int
        """
        super().__init__()
        self.model = model
        self.modality_dimensions = modality_dimensions
        self.add_feature_names = add_feature_names
        self.model.eval()
        logging.debug(self.model)

    def __call__(
        self,
        sample_dict: NDict,
        key_in=None,
        key_out=None,
        **kwargs,
    ) -> Union[None, dict, List[dict]]:
        """performs a forward pass on key_in and stores the output on key_out

        :param sample_dict: sample dictionary
        :type sample_dict: NDict
        :param key_in: input dictionary key, defaults to None
        :type key_in: str, optional
        :param key_out: dictionary key to store the output, defaults to None
        :type key_out: str, optional
        :raises ValueError: if the number of dimensions is not supported
        :return: updated sample dict
        :rtype: Union[None, dict, List[dict]]
        """

        if isinstance(sample_dict[key_in], np.ndarray):
            input_tensor = torch.from_numpy(sample_dict[key_in])
        else:
            input_tensor = sample_dict[key_in]

        while len(input_tensor.shape) < self.modality_dimensions + 1:
            input_tensor = input_tensor.unsqueeze(0)

        logging.debug(
            f"input tensor shape: {input_tensor.shape} and type: {input_tensor.dtype}"
        )

        # get sample id and input key
        input_tensor = input_tensor.float()
        logging.debug(type(input_tensor))
        output_tensor = self.model(input_tensor)
        sample_dict[key_out] = output_tensor.detach().squeeze()

        if self.add_feature_names:
            names_key_in = f"names.{key_in}"
            names_key_out = f"names.{key_out}"

            names_in = sample_dict.get(
                names_key_in, list(range(torch.numel(input_tensor)))
            )
            sample_dict[names_key_in] = names_in

            if torch.equal(output_tensor, input_tensor):
                sample_dict[names_key_out] = sample_dict[names_key_in]
            else:
                feature_names = []
                flat_key_out = key_out.replace(".", "_")
                for feature_n in range(torch.numel(output_tensor)):
                    feature_names.append(f"{flat_key_out}.feat_{feature_n}")
                sample_dict[names_key_out] = feature_names

        return sample_dict
