import pandas as pd
import random


def get_splits(pickle_path="splits_final.pkl", split_id=0):
    splits = pd.read_pickle(pickle_path)

    train_val_range = []
    for sample_id_str in splits[split_id]["train"]:
        sample_id = int(sample_id_str.split("_")[-1])
        train_val_range.append(sample_id)

    val_length = int(0.2 * len(train_val_range))
    val_ids = random.choices(train_val_range, k=val_length)
    train_ids = [v for v in train_val_range if v not in val_ids]

    test_ids = []
    for sample_id_str in splits[split_id]["val"]:
        sample_id = int(sample_id_str.split("_")[-1])
        test_ids.append(sample_id)

    data_splits = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}

    return data_splits


def get_splits_str_ids(pickle_path="splits_final.pkl", split_id=0):
    splits = pd.read_pickle(pickle_path)

    train_val_range = []
    for sample_id_str in splits[split_id]["train"]:
        train_val_range.append(sample_id_str)

    val_length = int(0.2 * len(train_val_range))
    val_ids = random.choices(train_val_range, k=val_length)
    train_ids = [v for v in train_val_range if v not in val_ids]

    test_ids = []
    for sample_id_str in splits[split_id]["val"]:
        test_ids.append(sample_id_str)

    data_splits = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}

    return data_splits
