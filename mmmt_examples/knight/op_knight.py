import json
import os
from typing import List, Union

import numpy as np
import pandas as pd
from fuse.data.ops.op_base import OpBase
from fuse.data.utils.sample import get_sample_id
from fuse.utils.ndict import NDict

CLINICAL_FEATURES_CHALLENGE = [
    "age_at_nephrectomy",
    "gender",
    "body_mass_index",
    "comorbidities",
    "smoking_history",
    "age_when_quit_smoking",
    "pack_years",
    "chewing_tobacco_use",
    "alcohol_use",
    "last_preop_egfr",
    "radiographic_size",
    "voxel_spacing",
]


class OpDecodeKNIGHT(OpBase):
    """
    Decoding operator class for the KNIGHT dataset. It has been reduced to
    minimal hardcoded values
    """

    def __init__(
        self,
        knight_data_path,
        categorical_threshold=10,
        selected_cols=None,
        label="aua_risk_group",
    ):
        super().__init__()

        # replacement values for non existing data points
        replacements = {
            "None": np.nan,
            "not_applicable": np.nan,
            None: np.nan,
            "<NA>": np.nan,
        }
        self.selected_cols = selected_cols
        self.label = label

        self.knight_data_path = os.path.abspath(knight_data_path)
        json_file = os.path.join(self.knight_data_path, "knight.json")
        with open(json_file) as file:
            json_content = json.load(file)

        # raw dataframe without processing
        self.raw_dataframe = pd.read_json(json_file)

        # preprocessing of the data, applying replacements and casting types
        # replacements of "non-values"
        dataframe = pd.json_normalize(json_content).replace(replacements)

        clinical_feature_list = []
        for col in dataframe.columns:
            for feature_name in CLINICAL_FEATURES_CHALLENGE:
                if feature_name in col:
                    clinical_feature_list.append(col)

        self.label_dataframe = dataframe[[self.label]]
        dataframe = dataframe[clinical_feature_list]

        # identify categorical columns
        for col in dataframe.columns:
            if len(dataframe[col].unique()) < categorical_threshold:
                dataframe[col] = dataframe[col].astype("category")

        # convert categorical columns to one-hot-encoded features
        for k in dataframe.select_dtypes(include="category").columns:
            dataframe = pd.concat(
                [dataframe, pd.get_dummies(dataframe[k], prefix=k)],
                axis=1,
                join="inner",
            )
        # convert non-categorical values to numbers (except case_id)
        for k in dataframe.select_dtypes(exclude="category").columns:
            if k != "case_id":
                dataframe[k] = pd.to_numeric(dataframe[k], errors="coerce")

        # only numerical features and one-hot encoded categories are kept
        self.dataframe = dataframe.select_dtypes("number").fillna(0)

    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:

        raw_key = "data.raw_clinical"
        clinical_key = "data.input.clinical_features"
        image_path_key = "data.input.image_path"
        label_key = "data.gt.label"
        clinical_names_key = "names.data.input.clinical_features"

        sid = get_sample_id(sample_dict)
        sample_dict[raw_key] = self.raw_dataframe.iloc[sid]
        sample_dict[clinical_key] = self.dataframe.iloc[sid].to_numpy(dtype=np.float32)
        sample_dict[image_path_key] = os.path.join(
            self.knight_data_path,
            sample_dict[raw_key]["case_id"],
            "imaging.nii.gz",
        )
        sample_dict[label_key] = int(
            self.label_dataframe.iloc[sid]["aua_risk_group"]
            in ["high_risk", "very_high_risk"]
        )
        sample_dict[clinical_names_key] = list(self.dataframe.columns)

        return sample_dict
