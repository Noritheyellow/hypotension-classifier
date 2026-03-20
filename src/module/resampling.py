import os
import yaml
import h5py
import argparse
import asyncio
import datetime
import numpy as np
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import module.utils as utils


async def process(file_path, fs, trks, save_path=None):
    caseid = file_path.name.split(".")[0]

    try:
        vf = await utils.read_vitalfile(file_path, trks)
    except Exception as e:
        print(f"{caseid}: {e}")
        return None

    ts, sig = utils.read_best_signal(vf, fs)
    ts_strings = np.array([t.isoformat() for t in ts.reshape(-1)], dtype="S26")
    sig_len = len(ts)
    cid = np.array([caseid] * sig_len)
    data = np.concatenate([cid.reshape(-1, 1), ts, sig], axis=1)

    if save_path is None:
        return data

    os.makedirs(save_path / caseid, exist_ok=True)
    save_name = f"{caseid}_{fs}fs.h5"
    try:
        with h5py.File(save_path / caseid / save_name, "w") as f:
            f.create_dataset("signal/ts", data=ts_strings, compression="gzip")
            f.create_dataset("signal/abp", data=sig.reshape(-1), compression="gzip")
            f["signal"].attrs["fs"] = fs
            f["signal"].attrs["len"] = sig_len
            f.attrs["caseid"] = caseid
    except Exception as e:
        print("Saving error: {e}")

    return data


async def async_process(file_path, fs, trks, save_path):
    tasks = [asyncio.create_task(process(fp, fs, trks, save_path)) for fp in file_path]
    result = list(filter(lambda x: x is not None, await tqdm_asyncio.gather(*tasks)))
    result = np.concatenate(result, axis=0)
    return result
