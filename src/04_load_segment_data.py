from pathlib import Path
from functools import partial
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
    src_path = Path(config["data_path"]).expanduser() / "02_04.processed"

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


def run(file_path, config, strategy: LabelStrategy, label: str):
    cid = str(file_path).split("/")[-2]
    case_path = Path(config["data_path"]).expanduser() / "02_04.processed" / cid
    file_name = str(file_path).split("/")[-1].split(".")[0][:-7] + "_segments.npy"
    event = load_npy(file_path)
    segment = strategy.extract_segment(event, config, label)
    save_npy(case_path / file_name, segment)
    return segment


# Usage: python src/04_load_segment_data.py --strategy hypophet
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
