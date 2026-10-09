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
    src_path = data_path / "mover" / "02.processed"
    save_path = data_path / "mover" / "05.dataset"
    file_path = sorted(src_path.rglob(f"*{config['fs']}fs*processed.h5"))
    mover_path = data_path / "mover" / "mover" / "mover-download.ics.uci.edu"
    cohort_path = (
        data_path
        / "mover"
        / "mover"
        / "mover-download.ics.uci.edu"
        / "EPIC_EMR"
        / "EMR"
    )
    env = {
        "src_path": src_path,
        "save_path": save_path,
        "file_path": file_path,
        "mover_path": mover_path,
        "cohort_path": cohort_path,
    }
    return env


def prepare_cohort(env) -> pd.DataFrame:
    pat_info = pd.read_csv(env["cohort_path"] / "patient_information.csv")
    pid_to_mrn = pd.read_csv(env["mover_path"] / "EPIC_MRN_PAT_ID.csv")
    cohort = (
        pd.merge(pid_to_mrn, pat_info, how="inner", on=["LOG_ID", "MRN"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # dept가 없어서 해당 처리는 하지 못하였다.
    cohort = (
        cohort.query("18 <= BIRTH_DATE < 65")
        .query("PRIMARY_ANES_TYPE_NM == 'General'")
        .reset_index(drop=True)
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
    mover_path = file_path.parent.parent.parent
    cid, _, _, ts, sig, _, _ = load_h5(file_path)
    cid = str(cid, "utf-8")
    base_name = (
        mover_path
        / "04.segment"
        / cid
        / f"{cid}_100fs_processed_{args.strategy}_adj{config['adjacency']}"
    )

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


# Usage: python src/05_load_data_mover.py --strategy hypophetversion2 -m 0 -M 300 -p 0
def main():
    """
    cohort 조건에 부합하는 사람들만으로 데이터셋을 구성한다.
    """
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(config)

    cohort = prepare_cohort(env)

    cohort_caseid = cohort.PAT_ID.values
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
