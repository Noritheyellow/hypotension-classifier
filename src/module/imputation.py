import datetime
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import yaml
import numpy as np
from tqdm import tqdm
import scipy.signal as signal
import matplotlib.pyplot as plt
import src.module.utils as utils


def impute(file_path, save_path=None):
    cid, ts, sig = np.load(file_path, allow_pickle=True).T
    sig = sig.astype(np.float32)
    try:
        sig = utils.carry_forward(sig)
        data = np.concatenate(
            [cid.reshape(-1, 1), ts.reshape(-1, 1), sig.reshape(-1, 1)], axis=1
        )
        np.save(
            # save_path / cid[0] / f"{'.'.join(file_path.name.split('.')[:-1])}_cf.npy",
            save_path / cid[0] / f"{'.'.join(file_path.name.split('.')[:-1])}_pps.npy",
            data,
        )

    except Exception as e:
        print(f"{cid[0]}: {e}")
        return None

    return data


def process_impute(sig):
    try:
        sig = utils.carry_forward(sig)
    except Exception as e:
        print(f"Imputation error: {e}")
        sig = None

    return sig


def main(mode, method, conf, src, dest):
    conf_path, src_path, save_path = Path(conf), Path(src), Path(dest)
    with open(conf_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if mode == "full":
        file_path = sorted(list(src_path.rglob(f"*{config['fs']}fs.npy")))

    elif mode == "part":
        if False:
            tar_conf = config["target"]
            sample_type = f"in{config['input_len']}out{config['pred_len']}"
            hypo_type = (
                f"pos{tar_conf['pos']}s{tar_conf['pos_time']}g{tar_conf['group_range']}"
            )
            if method == 2:
                hypo_type = (
                    f"{hypo_type}str{tar_conf['stride']}nt{tar_conf['nan_threshold']}"
                )
            normo_type = f"neg{tar_conf['neg']}s{tar_conf['neg_time']}j{tar_conf['adjacency']}g{tar_conf['group_range']}"

        else:
            tar_conf = config["target2"]
            sample_type = f"in{config['input_len']}out{config['pred_len']}"
            hypo_type = f"pos{tar_conf['pos']}s{tar_conf['pos_time']}str{tar_conf['pos_stride']}g{tar_conf['group_range']}seg{tar_conf['seg_stride']}segd{tar_conf['seg_density']}"
            normo_type = f"neg{tar_conf['pos']}j{tar_conf['adjacency']}"

        hypo_path = list(
            save_path.rglob(f"*{sample_type}*{hypo_type}*{config['fs']}fs.npy")
        )
        normo_path = list(
            save_path.rglob(f"*{sample_type}*{normo_type}*{config['fs']}fs.npy")
        )

        file_path = sorted(hypo_path + normo_path)

    starttime = datetime.datetime.now()
    pool = mp.Pool(processes=4)
    configured_main = partial(impute, config=config, save_path=save_path)
    # Create a tqdm progress bar
    with tqdm(total=len(file_path)) as pbar:
        for _ in pool.imap_unordered(configured_main, file_path):
            pbar.update(1)  # Update the progress bar for each completed task
    pool.close()
    pool.join()
    endtime = datetime.datetime.now()

    print("\nElapsed time: {}".format(endtime - starttime))


if __name__ == "__main__":
    """
    Description:
        imputation.py 는 `src` 로부터 `*fs.npy` 파일을 읽는다.
        본 작업은 이어지는 필터링의 연산 과정이 정상적으로 동작하게 하기 위한 사전작업이다.
        보간을 위해서는 직전 과거의 값을 사용하는 Carry forwarding 을 적용하였다.
        결과물은 `dest` 에 `*cf.npy` 로 저장된다.

    Command line:
        1) python src/v02/imputation.py -m full -c config/conf_v02.yaml -s dataset/v02/full_preprocess -d dataset/v02/full_preprocess
        2) python src/v02/imputation.py -m part -c config/conf_v02.yaml -s dataset/v02/part_preprocess -d dataset/v02/part_preprocess
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", dest="mode", type=str, help="select mode: [full, part]"
    )
    parser.add_argument(
        "--method", dest="method", type=int, help="select method: [1, 2]"
    )
    parser.add_argument("-c", "--conf", dest="conf", type=str)
    parser.add_argument("-s", "--src", dest="src", type=str)
    parser.add_argument("-d", "--dest", dest="dest", type=str)
    args = vars(parser.parse_args())

    main(**args)
