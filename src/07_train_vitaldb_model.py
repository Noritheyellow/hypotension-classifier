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
from sktime.classification.deep_learning import (
    InceptionTimeClassifier,
    CNNClassifier,
    LSTMFCNClassifier,
    ResNetClassifier,
)
from sklearn.metrics import roc_auc_score, classification_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

np.random.seed(42)


def set_environment(args, config, model_name) -> dict:
    data_path = Path(config["data_path"]).expanduser()
    src_path = data_path / "06.train_val_test"

    save_dir = Path(config["model_path"]).expanduser()
    save_dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = save_dir / (save_dt + f"_{args.strategy}") / f"{model_name}.model.keras"
    env = {"src_path": src_path, "save_path": save_path}
    return env


def balance_dataset(X, y):
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos_idx) > len(neg_idx):
        pos_idx = np.random.choice(pos_idx, size=len(neg_idx), replace=False)
    else:
        neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)

    pos_X, pos_y = X[pos_idx], y[pos_idx]
    neg_X, neg_y = X[neg_idx], y[neg_idx]
    res_X = np.vstack([pos_X, neg_X])
    res_y = np.vstack([pos_y, neg_y])
    print(res_y.shape)
    logging.info(
        f"X: {res_X.shape} / y(pos/neg): {res_y.size}({sum(res_y==1)[0]}/{sum(res_y==0)[0]})"
    )
    return res_X, res_y


def init_callbacks(env) -> list:
    callbacks = [
        ModelCheckpoint(
            env["save_path"],
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", factor=0.1, patience=10, mode="max", verbose=1
        ),
        EarlyStopping(
            monitor="val_auc", patience=15, mode="max", start_from_epoch=15, verbose=1
        ),
    ]
    return callbacks


def init_model(config, model_name) -> Model:
    if model_name == "inceptiontime":
        clf = InceptionTimeClassifier(
            kernel_size=40,
            n_filters=32,
            use_residual=True,
            use_bottleneck=True,
            bottleneck_size=32,
            depth=2,
            random_state=config["seed"],
        )

    elif model_name == "cnn":
        clf = CNNClassifier(
            kernel_size=7, batch_size=32, n_conv_layers=2, random_state=config["seed"]
        )

    elif model_name == "lstmfcn":
        clf = LSTMFCNClassifier(
            dropout=0.8,
            kernel_sizes=(8, 5, 3),
            filter_sizes=(128, 256, 128),
            lstm_size=8,
        )

    elif model_name == "resnet":
        clf = ResNetClassifier(random_state=42)

    return clf


def replace_head(model, input_shape, n_classes) -> Model:
    keras_model = model.build_model(input_shape=input_shape, n_classes=n_classes)
    x = keras_model.layers[-2].output
    output = Dense(1, activation="sigmoid")(x)
    new_model = Model(inputs=keras_model.input, outputs=output)
    new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["auc"])
    return new_model


def train_model(model, train_X, train_y, val_X, val_y, callbacks):
    print(model.summary())
    history = model.fit(
        train_X,
        train_y,
        validation_data=(val_X, val_y),
        epochs=1000,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


def evaluate_model(model, test_X, test_y, threshold):
    prob_y = model.predict(test_X)
    pred_y = np.where(prob_y >= threshold, 1, 0)
    print(roc_auc_score(test_y, prob_y))
    print(classification_report(test_y, pred_y))


# Usage: python src/07_train_vitaldb_model.py --strategy hypophetversion2
def main():
    model_name = "inceptiontime"
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config, model_name)
    train_X = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_train_X.npy")
    train_y = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_train_y.npy")
    val_X = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_val_X.npy")
    val_y = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_val_y.npy")
    test_X = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_test_X.npy")
    test_y = load_npy(env["src_path"] / f"vitaldb_{args.strategy}_test_y.npy")

    train_bal_X, train_bal_y = balance_dataset(train_X, train_y)
    print(train_X.shape, train_y.shape)
    print(train_bal_X.shape, train_bal_y.shape)
    print(val_X.shape, val_y.shape)
    print(test_X.shape, test_y.shape)

    callbacks = init_callbacks(env)

    clf = init_model(config, model_name)
    if model_name == "inceptiontime":
        clf = replace_head(clf, train_bal_X.shape[1:], 2)

    clf, history = train_model(clf, train_bal_X, train_bal_y, val_X, val_y, callbacks)
    evaluate_model(clf, test_X, test_y, 0.5)


if __name__ == "__main__":
    main()
