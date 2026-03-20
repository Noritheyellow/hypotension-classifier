import os
import datetime
import argparse
from pathlib import Path
import multiprocessing as mp
import yaml
import h5py
import numpy as np
from tqdm import tqdm
from functools import partial
from src.module.utils import (
    parse_args,
    load_config,
    process_meanbp,
    multiprocess_function,
)
from src.module.imputation import process_impute
from src.module.noise_filtering import process_filter, process_smooth
from src.module.signal_quality_assessment import process_assess


def set_environment(args, config):
    src_path = Path(config["output_path"]).expanduser() / "01.raw"
    save_path = Path(config["output_path"]).expanduser() / "02.processed"
    env = {"src_path": src_path, "save_path": save_path}
    return env


def save_to_hdf5(file_name, save_path, arr, fs, sig_len):
    cid, ts, sig, mbp, sqi = arr
    caseid = cid[0]
    ts_strings = ts.astype(np.float32)
    sig = sig.astype(np.float32)
    mbp = mbp.astype(np.float32)
    sqi = sqi.astype(np.float32)

    dir_name = save_path / caseid
    save_name = f"{file_name}_processed.h5"
    os.makedirs(dir_name, exist_ok=True)
    try:
        with h5py.File(dir_name / save_name, "w") as f:
            f.create_dataset("signal/ts", data=ts_strings, compression="gzip")
            f.create_dataset("signal/abp", data=sig, compression="gzip")
            f.create_dataset("signal/mbp", data=mbp, compression="gzip")
            f.create_dataset("signal/sqi", data=sqi, compression="gzip")
            f["signal"].attrs["fs"] = fs
            f["signal"].attrs["len"] = sig_len
            f.attrs["caseid"] = caseid.astype("S20")
    except Exception as e:
        print(f"Saving error: {e}")
        return 0

    return 1


def main(file_path, save_path, config):
    with h5py.File(file_path, "r") as f:
        caseid = f.attrs["id"]
        sig = f["signal"][:].reshape(-1)
    ts = np.empty_like(sig)
    cid = np.array([caseid] * len(sig))

    processed_sig = process_impute(sig)
    processed_sig = process_filter(processed_sig, config)
    processed_sig = process_smooth(processed_sig, config)
    meanbp = process_meanbp(processed_sig, config)
    sqi = process_assess(processed_sig, config)
    result = np.vstack([cid, ts, processed_sig, meanbp, sqi])  # (cols, timesteps)

    # 마지막 확장자 외에 다른 문자열 부분에 '.'이 존재하면 이건 파일명이므로 보존한다.
    fname = ".".join(file_path.name.split(".")[:-1])
    is_success = save_to_hdf5(fname, save_path, result, config["fs"], len(ts))
    return is_success


"""
* Usage: python src/02_transform_cleanse_data.py [options...]
*   -c, --conf              Set path for `conf.yaml`.
*   -m, --min               Set start index of wave files.
*   -M, --max               Set end index of wave files.
*   -p, --process           Set number of processor while multiprocessing.
"""
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    file_path = sorted(env["src_path"].rglob(f"*{config['fs']}fs.h5"))[
        args.min : args.max
    ]
    print("Total files: ", len(file_path))
    is_success = multiprocess_function(
        partial(main, save_path=env["save_path"], config=config),
        file_path,
        processes=args.process,
    )
    print(f"{sum(is_success)} files succeeded.")
