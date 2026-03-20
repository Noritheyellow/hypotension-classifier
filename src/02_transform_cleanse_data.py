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
from src.module.utils import load_config, process_meanbp
from src.module.imputation import process_impute
from src.module.noise_filtering import process_filter, process_smooth
from src.module.signal_quality_assessment import process_assess

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf", dest="conf", type=str)
parser.add_argument("-m", "--min", dest="min", type=int, default=None)
parser.add_argument("-M", "--max", dest="max", type=int, default=None)
args = parser.parse_args()


def save_to_hdf5(file_name, save_path, arr, fs, sig_len):
    cid, ts, sig, mbp, sqi = arr
    caseid = cid[0]
    ts_strings = np.array([t.isoformat() for t in ts], dtype="S26")
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
            f.attrs["caseid"] = caseid
    except Exception as e:
        print(f"Saving error: {e}")
        return 0

    return 1


def main(file_path, save_path, fs, config):
    with h5py.File(file_path, "r") as f:
        sig = f["signal/abp"][:].reshape(-1)
        ts = f["signal/ts"][:].astype("datetime64[us]").astype(object)
        caseid = f.attrs["caseid"]
    cid = np.array([caseid] * len(ts))

    processed_sig = process_impute(sig)
    processed_sig = process_filter(processed_sig, config)
    processed_sig = process_smooth(processed_sig, config)
    meanbp = process_meanbp(processed_sig, config)
    sqi = process_assess(processed_sig, config)
    result = np.vstack([cid, ts, processed_sig, meanbp, sqi])  # (cols, timesteps)

    # 마지막 확장자 외에 다른 문자열 부분에 '.'이 존재하면 이건 파일명이므로 보존한다.
    fname = ".".join(file_path.name.split(".")[:-1])
    is_success = save_to_hdf5(fname, save_path, result, fs, len(ts))
    return is_success


# Usage: python src/transform_cleanse_data.py -c config/conf.yaml -m 0 -M 10
if __name__ == "__main__":
    config = load_config(args.conf)

    src_path = Path(config["data_path"]).expanduser() / "01.raw"
    save_path = Path(config["data_path"]).expanduser() / "02_04.processed"
    file_path = sorted(list(src_path.rglob(f"*{config['fs']}fs.h5")))[
        args.min : args.max
    ]

    print(f"Configuration")
    print(f"- source path: {src_path}")
    print(f"- save path: {save_path}")
    print(f"- process imputation: carry-forwarding")
    print(f"- process filtering: chebyshev-II")
    print(f"- process smoothing: Moving average")
    print(f"- assess quality: [beat size, hgt, skewness, flatness, shape integrity]")
    print(f"- number of files: {len(file_path)}")

    is_success = []
    start_time = datetime.datetime.now()
    conf_main = partial(main, save_path=save_path, fs=config["fs"], config=config)
    pool = mp.Pool(processes=4)
    with tqdm(total=len(file_path)) as pbar:
        for x in pool.imap_unordered(conf_main, file_path):
            pbar.update(1)
            is_success.append(x)
    pool.close()
    pool.join()
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    print(f"{sum(is_success)} files succeeded.")
