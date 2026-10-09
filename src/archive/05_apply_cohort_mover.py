from functools import partial
import asyncio
from pathlib import Path
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from src.module.utils import parse_args, load_config, save_npy


def set_environment(args, config):
    src_path = Path(config["src_path"]).expanduser()
    output_path = Path(config["output_path"]).expanduser()
    cohort_path = src_path / "EPIC_EMR" / "EMR"
    test_path = output_path / "04.testset"
    save_path = output_path / "05.final"
    env = {
        "src_path": src_path,
        "cohort_path": cohort_path,
        "test_path": test_path,
        "save_path": save_path,
    }
    return env


def prepare_cohort(file_path) -> pd.DataFrame:
    cohort_path = file_path / "EPIC_EMR" / "EMR"
    cli_info = pd.read_csv(cohort_path / "patient_information.csv")
    pid_to_mrn = pd.read_csv(file_path / "EPIC_MRN_PAT_ID.csv")
    cohort = (
        pd.merge(pid_to_mrn, cli_info, how="inner", on=["LOG_ID", "MRN"])
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


def read_dataset(dir_path, fname_fmt) -> npt.NDArray:
    dataset = sorted(dir_path.rglob(fname_fmt))
    cid, segments = [], []
    for i in range(len(dataset)):
        with h5py.File(dataset[i], "r") as f:
            cid.append(f["caseid"][:].reshape(-1).astype(str))
            segments.append(f["segments"][:])

    cid = np.hstack(cid)
    segments = np.vstack(segments)
    print("cid: ", len(np.unique(cid)), " segments: ", segments.shape)
    return cid, segments


# Usage: python src/06_apply_cohort.py --strategy hypophet
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    cohort = prepare_cohort(env["src_path"])
    pos_cid, pos_segments = read_dataset(
        env["test_path"], f"testset_{args.strategy}_pos_segments*"
    )
    neg_cid, neg_segments = read_dataset(
        env["test_path"], f"testset_{args.strategy}_neg_segments*"
    )
    cohort_pos_cid = cohort.query("PAT_ID.isin(@pos_cid)").PAT_ID.unique()
    cohort_neg_cid = cohort.query("PAT_ID.isin(@neg_cid)").PAT_ID.unique()
    new_pos_cid = pos_cid[np.isin(pos_cid, cohort_pos_cid)]
    new_neg_cid = neg_cid[np.isin(neg_cid, cohort_neg_cid)]
    new_pos_seg = pos_segments[np.isin(pos_cid, cohort_pos_cid)]
    new_neg_seg = neg_segments[np.isin(neg_cid, cohort_neg_cid)]
    pos_y = np.ones(new_pos_cid.shape)
    neg_y = np.zeros(new_neg_cid.shape)
    print(
        "pos cid(unique): ",
        new_pos_cid.shape,
        len(cohort_pos_cid),
        " segments: ",
        new_pos_seg.shape,
    )
    print(
        "neg cid(unique): ",
        new_neg_cid.shape,
        len(cohort_neg_cid),
        " segments: ",
        new_neg_seg.shape,
    )

    X = np.vstack([new_pos_seg, new_neg_seg])
    y = np.hstack([pos_y, neg_y])
    print(X.shape, y.shape)

    save_npy(env["save_path"] / "mover_groundtruth_testset_X.npy", X)
    save_npy(env["save_path"] / "mover_groundtruth_testset_y.npy", y)


if __name__ == "__main__":
    main()
