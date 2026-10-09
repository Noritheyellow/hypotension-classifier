import os
import datetime
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import yaml
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.module.utils import (
    parse_args,
    load_config,
    load_npy,
    save_npy,
    multiprocess_function,
)
from src.module.prepare_dataset import preprocess, prepare


def set_environment(config):
    data_path = Path(config["data_path"]).expanduser()
    src_path = data_path / "vitaldb" / "02_04.processed"
    save_path = data_path / "vitaldb" / "05.dataset"
    file_path = sorted(src_path.rglob(f"*{config['fs']}fs*processed.h5"))
    cohort_path = data_path / "vitaldb" / "clinical_data.csv"
    env = {
        "src_path": src_path,
        "save_path": save_path,
        "file_path": file_path,
        "cohort_path": cohort_path,
    }
    return env


def prepare_cohort(file_path) -> pd.DataFrame:
    cohort = pd.read_csv(file_path).reset_index(drop=True)
    cohort["age_num"] = np.where((cohort["age"] == ">89"), 90, cohort["age"]).astype(
        np.float32  # age < 1
    )
    cohort = (
        cohort.query("18 <= age_num < 65")
        .drop_duplicates(["caseid"], keep=False)
        .query("department not in ['Gynecology']")
        # .query("department not in ['Thoracic surgery', 'Gynecology']")
        .query("ane_type == 'General'")
    )
    return cohort


def load_h5(file_path) -> tuple:
    with h5py.File(file_path, "r") as f:
        ts = f["signal/ts"][:]
        sig = f["signal/abp"][:]
        mbp = f["signal/mbp"][:]
        sqi = f["signal/sqi"][:]
        fs = f["signal"].attrs["fs"]
        sig_len = f["signal"].attrs["len"]
        cid = f.attrs["caseid"]

    return cid, fs, sig_len, ts, sig, mbp, sqi


def prepare_data(file_path, args, config):
    base_name = f"{str(file_path)[:-3]}_{args.strategy}_adj{config['adjacency']}"

    cid, _, _, ts, sig, _, _ = load_h5(file_path)
    try:
        pos_segments_idx = np.load(f"{base_name}_pred_pos_segments.npy")
        pos_segments_idx = np.vstack([np.arange(*s) for s in pos_segments_idx])
        pos_segments = np.expand_dims(sig[pos_segments_idx], axis=-1)
        pos_ts = np.expand_dims(ts[pos_segments_idx], axis=-1)
        pos_cid = np.full(pos_segments.shape, [cid], dtype="<U4")
        pos_data = np.concatenate([pos_cid, pos_ts, pos_segments], axis=-1, dtype=str)
    except Exception as e:
        pos_data = np.array([])

    try:
        neg_segments_idx = np.load(f"{base_name}_neg_segments.npy")
        neg_segments_idx = np.vstack([np.arange(*s) for s in neg_segments_idx])
        neg_segments = np.expand_dims(sig[neg_segments_idx], axis=-1)
        neg_ts = np.expand_dims(ts[neg_segments_idx], axis=-1)
        neg_cid = np.full(neg_segments.shape, [cid], dtype="<U4")
        neg_data = np.concatenate([neg_cid, neg_ts, neg_segments], axis=-1, dtype=str)
    except Exception as e:
        neg_data = np.array([])

    return pos_data, neg_data


# Usage: python src/05_load_data.py --strategy hypophetversion2 -m 0 -M 300 -p 0
def main():
    """
    cohort 조건에 부합하는 사람들만으로 데이터셋을 구성한다.
    """
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(config)

    cohort = prepare_cohort(env["cohort_path"])
    cohort_caseid = cohort.caseid.apply(lambda x: str(x).zfill(4)).values
    directories = np.array(os.listdir(env["src_path"]))
    cohort_dirs = directories[np.isin(directories, cohort_caseid)]
    cohort_file_path = list(
        filter(lambda x: str(x).split("/")[-2] in cohort_dirs, env["file_path"])
    )
    print(len(cohort_file_path))

    pos_dataset, neg_dataset = [], []
    custom_prepare_data = partial(prepare_data, args=args, config=config)
    for file_path in tqdm(cohort_file_path[args.min : args.max], ncols=75):
        pos_data, neg_data = custom_prepare_data(file_path)
        pos_dataset.append(pos_data)
        neg_dataset.append(neg_data)

    pos_dataset = np.vstack(list(filter(lambda x: len(x) > 0, pos_dataset)))
    neg_dataset = np.vstack(list(filter(lambda x: len(x) > 0, neg_dataset)))
    print("pos_dataset size: ", pos_dataset.shape)
    print("neg_dataset size: ", neg_dataset.shape)

    np.save(
        env["save_path"]
        / f"{args.strategy}_adj{config['adjacency']}_pos_dataset_{args.part}.npy",
        pos_dataset,
    )
    np.save(
        env["save_path"]
        / f"{args.strategy}_adj{config['adjacency']}_neg_dataset_{args.part}.npy",
        neg_dataset,
    )
    print("Datasets saved.")


if __name__ == "__main__":
    main()
