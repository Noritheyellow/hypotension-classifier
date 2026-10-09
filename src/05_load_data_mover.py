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
    save_path = data_path / "mover" / "05.dataset_new"
    file_path = sorted(src_path.rglob(f"*{config['fs']}fs*processed.h5"))
    mover_path = data_path / "mover"
    env = {
        "src_path": src_path,
        "save_path": save_path,
        "file_path": file_path,
        "mover_path": mover_path,
    }
    return env


def prepare_cohort(env) -> pd.DataFrame:
    pat_info = pd.read_csv(env["mover_path"] / "patient_information.csv")
    pat_code = pd.read_csv(env["mover_path"] / "patient_coding.csv")
    pid_to_mrn = pd.read_csv(env["mover_path"] / "EPIC_MRN_PAT_ID.csv")
    selected_mrn = pat_code[
        ~pat_code.REF_BILL_CODE.apply(lambda x: str(x)[:2] in ["02", "10", "0U"])
    ]["MRN"].unique()
    cohort = (
        pd.merge(pid_to_mrn, pat_info, how="inner", on=["LOG_ID", "MRN"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # MOVER = Exclude cardiac('02'), gynecologic('10', '0U')
    cohort = (
        cohort.query("18 <= BIRTH_DATE")
        # .query("SEX == 'Male'")
        .query("PRIMARY_ANES_TYPE_NM == 'General'")
        .query("MRN.isin(@selected_mrn)")
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
    input_len = config["fs"] * config["input_len"]
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
        pos_segments = np.expand_dims(sig[pos_segments_idx], axis=-1).reshape(
            -1, input_len
        )
        pos_ts = np.expand_dims(ts[pos_segments_idx], axis=-1).reshape(-1, input_len)
        pos_cid = np.repeat(cid, len(pos_segments))
        pos_lt = np.load(f"{base_name}_pred_pos_lead_times.npy")
    except Exception as e:
        pos_segments = np.array([]).reshape(-1, input_len)
        pos_ts = np.array([]).reshape(-1, input_len)
        pos_cid = np.array([])
        pos_lt = np.array([])

    try:
        neg_segments_idx = np.load(f"{base_name}_neg_segments.npy")
        neg_segments = np.expand_dims(sig[neg_segments_idx], axis=-1).reshape(
            -1, input_len
        )
        neg_ts = np.expand_dims(ts[neg_segments_idx], axis=-1).reshape(-1, input_len)
        neg_cid = np.repeat(cid, len(neg_segments))
    except Exception as e:
        neg_segments = np.array([]).reshape(-1, input_len)
        neg_ts = np.array([]).reshape(-1, input_len)
        neg_cid = np.array([])

    return (pos_segments, pos_ts, pos_cid, pos_lt), (neg_segments, neg_ts, neg_cid)


# Usage: python src/05_load_data_mover.py --strategy hypophetversion2 -m 0 -M 300
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

    pos_segments, pos_ts, pos_cid, pos_lt = [], [], [], []
    neg_segments, neg_ts, neg_cid = [], [], []
    custom_prepare_data = partial(prepare_data, args=args, config=config)
    for file_path in tqdm(cohort_file_path[args.min : args.max], ncols=75):
        pos_data, neg_data = custom_prepare_data(file_path)
        pos_segments.append(pos_data[0])
        pos_ts.append(pos_data[1])
        pos_cid.append(pos_data[2])
        pos_lt.append(pos_data[3])
        neg_segments.append(neg_data[0])
        neg_ts.append(neg_data[1])
        neg_cid.append(neg_data[2])

    pos_segments = np.vstack(pos_segments, dtype=np.float32)
    pos_ts = np.vstack(pos_ts)
    pos_cid = np.hstack(pos_cid)
    pos_lt = np.hstack(pos_lt)
    neg_segments = np.vstack(neg_segments, dtype=np.float32)
    neg_ts = np.vstack(neg_ts)
    neg_cid = np.hstack(neg_cid)
    print(
        "pos_dataset size: ",
        pos_segments.shape,
        pos_ts.shape,
        pos_cid.shape,
        pos_lt.shape,
    )
    print("neg_dataset size: ", neg_segments.shape, neg_ts.shape, neg_cid.shape)

    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_pos_segments.npy",
        pos_segments,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_pos_ts.npy",
        pos_ts,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_pos_cid.npy",
        pos_cid,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_pos_lt.npy",
        pos_lt,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_neg_segments.npy",
        neg_segments,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_neg_ts.npy",
        neg_ts,
    )
    np.save(
        env["save_path"] / f"{args.strategy}_adj{config['adjacency']}_neg_cid.npy",
        neg_cid,
    )
    print("Datasets saved.")


if __name__ == "__main__":
    main()
