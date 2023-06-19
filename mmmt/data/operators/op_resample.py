from typing import List, Union

import torch.nn.functional as F
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
import numpy as np
import torch


class Op3DResample(OpBase):
    """
    Resampler of 3D tensors to desired size
    """

    def __init__(self, desired_size=[16, 16, 16], mode="nearest"):
        """constructor method

        :param desired_size: desired size of the tensor, defaults to [16, 16, 16]
        :type desired_size: list, optional
        :param mode: interpolation method, defaults to "nearest"
        :type mode: str, optional
        """
        super().__init__()

        self.desired_size = desired_size
        self.mode = mode

    def __call__(
        self,
        sample_dict: NDict,
        key_in="data.input.img",
        key_out="data.input.img",
        **kwargs,
    ) -> Union[None, dict, List[dict]]:
        """performs the resampling of a sample

        :param sample_dict: sample dictionary
        :type sample_dict: NDict
        :param key_in: input dictionary key, defaults to "data.input.img"
        :type key_in: str, optional
        :param key_out: output dictionary key, defaults to "data.input.img"
        :type key_out: str, optional
        :return: updated sample dict
        :rtype: Union[None, dict, List[dict]]
        """

        dimensions = len(self.desired_size)
        if dimensions == 1:
            print("Only one dimension")
            new_size = [self.desired_size for dim in sample_dict[key_in].shape]
            self.desired_size = new_size

        input = sample_dict[key_in]
        if type(input) == np.ndarray:
            input = torch.from_numpy(input)

        sample_dict[key_out] = F.interpolate(
            sample_dict[key_in].unsqueeze(0).unsqueeze(0),
            self.desired_size,
            mode=self.mode,
        ).squeeze()

        return sample_dict
