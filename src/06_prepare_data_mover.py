# Desc:
#   - 06.apply_cohort.py 출력물을 이용해서 모델을 학습시킨다.
#
# Output:
#   - 학습된 모델
#   - 평가에 사용할 Case IDs (12.test_vitaldb_model_with_vitaldb.py 에서 사용)
import datetime
from pathlib import Path
import numpy as np
import numpy.typing as npt
from src.module.utils import parse_args, load_config, load_npy, save_npy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

from keras.models import Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sktime.classification.deep_learning import InceptionTimeClassifier
from sklearn.metrics import roc_auc_score, classification_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

np.random.seed(42)


def set_environment(args, config) -> dict:
    data_path = Path(config["data_path"]).expanduser()
    src_path = data_path / "mover" / "05.dataset_new"
    pos_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_pos_*.npy"))
    )
    neg_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_neg_*.npy"))
    )

    env = {"src_path": src_path, "pos_path": pos_path, "neg_path": neg_path}
    return env


def load_dataset(env: dict) -> tuple[npt.NDArray, npt.NDArray]:
    pos_cid = load_npy(env["pos_path"][0])
    neg_cid = load_npy(env["neg_path"][0])
    # env['pos_path'] : pos_lt
    pos_seg = load_npy(env["pos_path"][2])
    neg_seg = load_npy(env["neg_path"][1])
    pos_ts = load_npy(env["pos_path"][3])
    neg_ts = load_npy(env["neg_path"][2])
    pos_y, neg_y = np.ones(len(pos_seg)), np.zeros(len(neg_seg))
    X, y = np.vstack([pos_seg, neg_seg]), np.hstack([pos_y, neg_y])
    cid = np.hstack([pos_cid, neg_cid])
    ts = np.vstack([pos_ts, neg_ts])
    logging.info(f"load_dataset(pos/neg): {y.size}({pos_y.size}/{neg_y.size})")
    print(X.shape, y.shape, cid.shape, ts.shape)
    return cid, ts, X, y


# Usage: python src/06_prepare_data_mover.py --strategy hypophetversion2
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    cid, ts, X, y = load_dataset(env)
    # X = scale_input(X)
    print(cid.shape, X.shape, y.shape, sum(y == 1), sum(y == 0))

    save_path = env["src_path"].parent / "06.train_val_test_new"
    save_npy(save_path / f"mover_{args.strategy}_test_cid.npy", cid)
    save_npy(save_path / f"mover_{args.strategy}_test_X.npy", X)
    save_npy(save_path / f"mover_{args.strategy}_test_y.npy", y)
    print("Dataset saved.")


if __name__ == "__main__":
    main()
