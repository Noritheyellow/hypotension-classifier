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
    src_path = data_path / "mover" / "05.dataset"
    pos_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_pos_dataset*"))
    )
    neg_path = np.array(
        sorted(src_path.rglob(f"{args.strategy}_adj{config['adjacency']}_neg_dataset*"))
    )

    env = {"src_path": src_path, "pos_path": pos_path, "neg_path": neg_path}
    return env


def load_dataset(env: dict) -> tuple[npt.NDArray, npt.NDArray]:
    pos_ds = np.vstack([load_npy(f) for f in tqdm(env["pos_path"], ncols=75)])
    neg_ds = np.vstack([load_npy(f) for f in tqdm(env["neg_path"][:1], ncols=75)])
    pos_y, neg_y = np.ones((len(pos_ds), 1)), np.zeros((len(neg_ds), 1))
    ds, y = np.vstack([pos_ds, neg_ds]), np.vstack([pos_y, neg_y])
    cid, dt, X = ds.transpose(2, 0, 1)
    logging.info(f"load_dataset(pos/neg): {y.size}({pos_y.size}/{neg_y.size})")
    return cid, dt, X, y


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
        f"X: {extract_x.shape} / y(pos/neg): {extract_y.size}({sum(extract_y==1)[0]}/{sum(extract_y==0)[0]})"
    )
    return extract_x, extract_y


# Usage: python src/06_prepare_data_mover.py --strategy hypophetversion2
def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    cid, dt, X, y = load_dataset(env)
    X = scale_input(X)

    print(X.shape, y.shape)

    # train_cid, val_cid, test_cid = split_array(np.unique(cid), 0.2, config["seed"])
    # train_X, train_y = extract_dataset(X, y, np.isin(cid[:, 0], train_cid))
    # val_X, val_y = extract_dataset(X, y, np.isin(cid[:, 0], val_cid))
    # test_X, test_y = extract_dataset(X, y, np.isin(cid[:, 0], test_cid))

    save_path = env["src_path"].parent / "06.train_val_test"
    save_npy(save_path / f"mover_{args.strategy}_test_X.npy", X)
    save_npy(save_path / f"mover_{args.strategy}_test_y.npy", y)
    # save_npy(save_path / f"vitaldb_{args.strategy}_train_X.npy", train_X)
    # save_npy(save_path / f"vitaldb_{args.strategy}_train_y.npy", train_y)
    # save_npy(save_path / f"vitaldb_{args.strategy}_val_X.npy", val_X)
    # save_npy(save_path / f"vitaldb_{args.strategy}_val_y.npy", val_y)
    # save_npy(save_path / f"vitaldb_{args.strategy}_test_X.npy", test_X)
    # save_npy(save_path / f"vitaldb_{args.strategy}_test_y.npy", test_y)
    print("Dataset saved.")


if __name__ == "__main__":
    main()
