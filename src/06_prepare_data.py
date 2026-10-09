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


def set_environment(args, config, model_name) -> dict:
    data_path = Path(config["data_path"]).expanduser()
    src_path = data_path / "vitaldb" / "05.dataset_new"
    pos_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_pos_*.npy"))
    )
    neg_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_neg_*.npy"))
    )

    save_dir = Path(config["model_path"]).expanduser()
    save_dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = save_dir / (save_dt + f"_{args.strategy}") / f"{model_name}.model.keras"
    env = {
        "src_path": src_path,
        "pos_path": pos_path,
        "neg_path": neg_path,
        "save_path": save_path,
    }
    return env


def load_dataset(env: dict) -> tuple[npt.NDArray, npt.NDArray]:
    pos_cid = load_npy(env["pos_path"][0])
    neg_cid = load_npy(env["neg_path"][0])
    pos_seg = load_npy(env["pos_path"][1])
    neg_seg = load_npy(env["neg_path"][1])
    pos_ts = load_npy(env["pos_path"][2])
    neg_ts = load_npy(env["neg_path"][2])
    pos_y, neg_y = np.ones(len(pos_seg)), np.zeros(len(neg_seg))
    X, y = np.vstack([pos_seg, neg_seg]), np.hstack([pos_y, neg_y])
    cid = np.hstack([pos_cid, neg_cid])
    ts = np.vstack([pos_ts, neg_ts])
    logging.info(f"load_dataset(pos/neg): {y.size}({pos_y.size}/{neg_y.size})")
    print(X.shape, y.shape, cid.shape, ts.shape)
    return cid, ts, X, y


def scale_input(input) -> npt.NDArray:
    return input.astype(np.float32) / 200


def split_array(arr, ratio, seed) -> tuple:
    train, test = train_test_split(arr, test_size=ratio, random_state=seed)
    train, val = train_test_split(train, test_size=ratio, random_state=seed)
    logging.info(
        f"total(train/val/test): {arr.size}({train.size}/{val.size}/{test.size}) cases"
    )
    return train, val, test


def extract_dataset(x, y, idx):
    extract_x = np.expand_dims(x[idx], 2)
    extract_y = y[idx]
    print(extract_y.shape)
    logging.info(
        f"X: {extract_x.shape} / y(pos/neg): {extract_y.size}({sum(extract_y==1)}/{sum(extract_y==0)})"
    )
    return extract_x, extract_y


# Usage: python src/06_prepare_data.py --strategy hypophetversion2
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config, "inceptiontime")
    cid, ts, X, y = load_dataset(env)
    # X = scale_input(X)
    print(X.shape, y.shape)

    train_cid, val_cid, test_cid = split_array(np.unique(cid), 0.2, config["seed"])
    train_X, train_y = extract_dataset(X, y, np.isin(cid, train_cid))
    val_X, val_y = extract_dataset(X, y, np.isin(cid, val_cid))
    test_X, test_y = extract_dataset(X, y, np.isin(cid, test_cid))

    print(
        train_cid.shape,
        train_X.shape,
        train_y.shape,
        sum(train_y == 1),
        sum(train_y == 0),
    )
    print(val_cid.shape, val_X.shape, val_y.shape, sum(val_y == 1), sum(val_y == 0))
    print(
        test_cid.shape, test_X.shape, test_y.shape, sum(test_y == 1), sum(test_y == 0)
    )
    save_path = env["src_path"].parent / "06.train_val_test_new"
    save_npy(save_path / f"vitaldb_{args.strategy}_train_cid.npy", train_cid)
    save_npy(save_path / f"vitaldb_{args.strategy}_train_X.npy", train_X)
    save_npy(save_path / f"vitaldb_{args.strategy}_train_y.npy", train_y)
    save_npy(save_path / f"vitaldb_{args.strategy}_val_cid.npy", val_cid)
    save_npy(save_path / f"vitaldb_{args.strategy}_val_X.npy", val_X)
    save_npy(save_path / f"vitaldb_{args.strategy}_val_y.npy", val_y)
    save_npy(save_path / f"vitaldb_{args.strategy}_test_cid.npy", test_cid)
    save_npy(save_path / f"vitaldb_{args.strategy}_test_X.npy", test_X)
    save_npy(save_path / f"vitaldb_{args.strategy}_test_y.npy", test_y)
    print("Dataset saved.")


if __name__ == "__main__":
    main()
