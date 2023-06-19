from typing import List, Union, Sequence

from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict


class OpConcatNames(OpBase):
    """
    Concatenate feature names. When two keys containing feature vectors are concatenated, the names
    of the features also need to be concatenated. Like this:
    [clinical1... clinicalN] + [imagingM... imagingM] produce
    [clinical1... clinicalN, imaging1... imagingM]
    """

    def __call__(
        self, sample_dict: NDict, keys_in: Sequence[str], key_out: str
    ) -> Union[None, dict, List[dict]]:
        """performs the concatenation of N lists of names

        :param sample_dict: sample dict
        :type sample_dict: NDict
        :param keys_in: dictionary keys that contain names that will be concatenated
        :type keys_in: Sequence[str]
        :param key_out: new dictionary key that will contain the concatenated names
        :type key_out: str
        :return: updated sample dict
        :rtype: Union[None, dict, List[dict]]
        """

        feature_names = []
        for concat_key in keys_in:
            names_key_in = f"names.{concat_key}"
            feature_names = feature_names + sample_dict[names_key_in]

        sample_dict[key_out] = feature_names

        return sample_dict
