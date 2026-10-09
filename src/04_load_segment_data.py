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

# 01~04까지의 저장 폴더는 hypophet_v2 쪽과 공유.


def set_environment(args, config):
    src_path = Path(config["data_path"]).expanduser() / "vitaldb" / "02_04.processed"

    ##### NEW #####
    pos_event_path = sorted(src_path.rglob(f"*{args.strategy}*_pred_pos_events.npy"))
    pos_event_path = [x for x in pos_event_path if not x.name.startswith("._")][
        args.min : args.max
    ]
    neg_event_path = sorted(src_path.rglob(f"*{args.strategy}*_neg_events.npy"))
    neg_event_path = [x for x in neg_event_path if not x.name.startswith("._")][
        args.min : args.max
    ]
    ###############
    env = {
        "src_path": src_path,
        "pos_event_path": pos_event_path,
        "neg_event_path": neg_event_path,
    }
    return env


def range_segment(segments):
    if segments.size != 0:
        ranges = np.apply_along_axis(lambda x: np.arange(*x), axis=1, arr=segments)
    else:
        ranges = np.array([], dtype=np.int32)
    return ranges


def assess_segment(ranges, sqi, config):
    if len(ranges) != 0:
        is_bad_quality = np.any(sqi[ranges] > config["sqi"]["threshold"], axis=1)
        samples = ranges[~is_bad_quality]
    else:
        samples = np.array([], dtype=np.int32).reshape(
            -1, config["fs"] * config["input_len"]
        )
    return samples


def run(file_path, config, strategy: LabelStrategy, label: str):
    cid = str(file_path).split("/")[-2]
    case_path = (
        Path(config["data_path"]).expanduser() / "vitaldb" / "02_04.processed" / cid
    )
    file_name = str(file_path).split("/")[-1].split(".")[0][:-7] + "_segments.npy"
    sig_path = case_path / f"{cid}_100fs_processed.h5"

    with h5py.File(sig_path, "r") as f:
        cid = f.attrs["caseid"]
        sqi = f["signal/sqi"][:].astype(np.float32)

    sqi = np.where(np.isnan(sqi), 1.0, sqi)
    events = load_npy(file_path)
    segments = strategy.extract_segment(events, config, label)
    ranges = range_segment(segments)
    samples = assess_segment(ranges, sqi, config)

    save_npy(case_path / file_name, samples)
    return samples


# Usage: python src/04_load_segment_data.py --strategy hypophetversion2
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    strategy = select_strategy(args.strategy)
    pos_segments = multiprocess_function(
        partial(run, config=config, strategy=strategy, label="pos"),
        env["pos_event_path"],
        8,
    )
    neg_segments = multiprocess_function(
        partial(run, config=config, strategy=strategy, label="neg"),
        env["neg_event_path"],
        8,
    )


if __name__ == "__main__":
    main()
