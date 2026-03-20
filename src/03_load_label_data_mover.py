import os
from pathlib import Path
from functools import partial
import h5py
import numpy as np
from src.module.utils import (
    parse_args,
    load_config,
    load_npy,
    save_npy,
    multiprocess_function,
    select_strategy,
)
from src.module.labelling import LabelStrategy


def set_environment(config):
    src_path = Path(config["data_path"]).expanduser() / "mover" / "02.processed"
    save_path = Path(config["data_path"]).expanduser() / "mover" / "03.label"
    file_path = sorted(src_path.rglob(f"*{config['fs']}fs*processed.h5"))
    env = {"src_path": src_path, "save_path": save_path, "file_path": file_path}
    return env


def load_h5(file_path) -> tuple:
    with h5py.File(file_path, "r") as f:
        ts = f["signal/ts"][:]
        sig = f["signal/abp"][:]
        mbp = f["signal/mbp"][:]
        sqi = f["signal/sqi"][:]
        fs = f["signal"].attrs["fs"]
        sig_len = f["signal"].attrs["len"]
        cid = f.attrs["caseid"].astype(str)

    return cid, fs, sig_len, ts, sig, mbp, sqi


def run(file_path, config, mode, strategy: LabelStrategy):
    input_len = config["input_len"]
    pred_len = config["pred_len"]
    grp = config["event_condition"]["pos"]["group_range"]
    adj = config["event_condition"]["neg"]["adjacency"]

    cid, _, _, _, _, mbp, sqi = load_h5(file_path)
    case_path = Path(config["data_path"]).expanduser() / "mover" / "03.label" / cid
    os.makedirs(case_path, exist_ok=True)
    fname = file_path.name.split(".")[0]
    pos_name = (
        f"{fname}_in{input_len}out{pred_len}_grp{grp}_{strategy.name}_pos_events.npy"
    )
    neg_name = (
        f"{fname}_in{input_len}out{pred_len}_adj{adj}_{strategy.name}_neg_events.npy"
    )

    ##### NEW #####
    base_name = f"{fname}_{strategy.name}_adj{adj}"
    ###############

    is_bad_quality = sqi > config["sqi"]["threshold"]
    masked_mbp = np.where(is_bad_quality, 0, mbp)

    if mode == 0:  # run all
        # hypophetversion2는 아래 코드를 사용하지 않으므로 수정 필요
        # pos_event = strategy.label_pos_event(masked_mbp, config)
        # neg_event = strategy.label_neg_event(masked_mbp, pos_event, config)
        (
            neg_event,
            actual_pos_event,
            pos_event,
            soft_gray_zone,
            hard_gray_zone,
        ) = strategy.label_event(masked_mbp, config)
        save_npy(case_path / f"{base_name}_neg_events.npy", neg_event)
        save_npy(case_path / f"{base_name}_actual_pos_events.npy", actual_pos_event)
        save_npy(case_path / f"{base_name}_pred_pos_events.npy", pos_event)
        save_npy(case_path / f"{base_name}_soft_gray_zones.npy", soft_gray_zone)
        save_npy(case_path / f"{base_name}_hard_gray_zones.npy", hard_gray_zone)

    elif mode == 1:  # run positive only
        pos_event = strategy.label_pos_event(masked_mbp, config)
        neg_event = load_npy(case_path / neg_name)
        save_npy(case_path / pos_name, pos_event)

    elif mode == 2:  # run negative only
        pos_event = load_npy(case_path / pos_name)
        neg_event = strategy.label_neg_event(masked_mbp, pos_event, config)
        save_npy(case_path / neg_name, neg_event)

    else:
        print("Wrong mode. You can choose 0, 1, 2.")

    return pos_event, neg_event


# Usage: python src/03_load_label_data_mover.py --strategy hypophetversion2 --mode 0
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(config)
    strategy = select_strategy(args.strategy)
    events = multiprocess_function(
        partial(run, config=config, mode=args.mode, strategy=strategy),
        env["file_path"][args.min : args.max],
        processes=8,
    )


"""
* Usage: python src/03_load_label_data.py [options...]
*   -c, --conf              Set path for `conf.yaml`.
*   -m, --min               Set start index of wave files.
*   -M, --max               Set end index of wave files.
*       --strategy          Set labelling strategy.
*       --mode              Set mode whether label both or not. (0: both, 1: positive, 2: negative)
*   -p, --process           Set number of processor while multiprocessing.
*
* Example: python src/03_load_label_data.py --strategy groundtruth --mode 0 -p 8
"""
if __name__ == "__main__":
    main()
