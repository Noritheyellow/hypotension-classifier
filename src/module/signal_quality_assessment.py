import datetime
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import yaml
import numpy as np
from tqdm import tqdm
import src.module.utils as utils
import warnings

# PeakPropertyWarning: some peaks have a prominence of 0
# 신호 중에 평평한 선이 존재하는 구간이 있는데 이 구간에 대해서 peak를 계산할 필요는 없으므로
# 이 경고를 무시하도록 설정.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def assess(file_path, config, save_path):
    cid, ts, sig = np.load(file_path, allow_pickle=True).T
    sig = sig.astype(np.float32)
    try:
        peaks, troughs = utils.get_peak_trough(sig, config["detection"])
        sqi = utils.assess_beat(
            sig,
            troughs,
            approx_zero=config["sqi"]["zero_interval"],
            thresholds=list(config["sqi"]["params"].values()),
        )
        data = np.concatenate(
            [
                cid.reshape(-1, 1),
                ts.reshape(-1, 1),
                sig.reshape(-1, 1),
                sqi.reshape(-1, 1),
            ],
            axis=1,
        )
        np.save(
            save_path / cid[0] / f"{'.'.join(file_path.name.split('.')[:-1])}.npy",
            data,
        )
    except Exception as e:
        print(f"{cid[0]}: {e}")
        return None

    return data


def process_assess(sig, config):
    try:
        _, troughs = utils.get_peak_trough(sig, config["detection"])
        sqi = utils.assess_beat(
            sig,
            troughs,
            config["sqi"]["zero_interval"],
            list(config["sqi"]["params"].values()),
        )
    except Exception as e:
        print(f"Assessing error: {e}")
        sqi = None

    return sqi


def main(mode, method, conf, src, dest):
    conf_path, src_path, save_path = Path(conf), Path(src), Path(dest)
    with open(conf_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if mode == "full":
        # file_path = sorted(list(save_path.rglob(f"*{config['fs']}fs*filt.npy")))
        file_path = sorted(list(save_path.rglob(f"*{config['fs']}fs*pps.npy")))

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
            # save_path.rglob(f"*{sample_type}*{hypo_type}*{config['fs']}fs*filt.npy")
            save_path.rglob(f"*{sample_type}*{hypo_type}*{config['fs']}fs*pps.npy")
        )
        normo_path = list(
            # save_path.rglob(f"*{sample_type}*{normo_type}*{config['fs']}fs*filt.npy")
            save_path.rglob(f"*{sample_type}*{normo_type}*{config['fs']}fs*pps.npy")
        )
        file_path = sorted(hypo_path + normo_path)

    starttime = datetime.datetime.now()
    pool = mp.Pool(processes=4)
    configured_main = partial(assess, config=config, save_path=save_path)
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
        signal_quality_assessment.py 는 `src` 로부터 `*filt.npy` 파일을 읽는다.
        본 코드는 각 신호의 Beat 단위로 품질 점수(SQI) 를 매기며 이 값은 0 ~ 1 사이의 값이다.
        신호의 품질을 평가하기 위해 다음의 다섯 가지를 평가한다:
            1) 비트의 크기
            2) 비트의 높이
            3) 비트값의 비대칭성
            4) 비트값의 평편도
            5) 비트의 중앙 기준 좌/우 형태 무결성(e.g. y축 반전 비트)

        허용된 기준 이내의 비트는 비대칭성과 평편도 점수의 합으로 계산되며 MinMaxScaling 으로 표준화된다.
        허용된 기준을 벗어나는 비트는 이상치로 간주하여 1 로 표기한다.
        결과물은 `dest` 에 `*sqi.npy` 로 저장된다.

    Command line:
        1) python src/v02/signal_quality_assessment.py -m full -c config/conf_v02.yaml -s dataset/v02/full_preprocess -d dataset/v02/full_preprocess
        2) python src/v02/signal_quality_assessment.py -m part -c config/conf_v02.yaml -s dataset/v02/part_preprocess -d dataset/v02/part_preprocess
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
