import asyncio
import argparse
from functools import partial
import multiprocessing as mp
import yaml
import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks
from tqdm import tqdm
from src.module.labelling import (
    Hypophet,
    HypophetVal1,
    HypophetVal2,
    HypophetVal3,
    HypophetTest1,
    HypophetTest2,
    HypophetTest3,
    HypophetTest4,
    HypophetTest5,
    HypophetTest6,
    HypophetTest7,
    HypophetTest8,
    HypophetVersion2,
    Acumen,
    Acumen2,
    Asan,
    Moghadam,
    GroundTruth,
    GroundTruth2,
)
import vitaldb


def pipeline(func_list):
    def flow(x):
        for func in func_list:
            x = func(x)
        return x

    return flow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="conf", type=str, default="config/conf.yaml")
    parser.add_argument("-b", dest="batch", type=int, default=100000)
    parser.add_argument("-m", "--min", dest="min", type=int, default=None)
    parser.add_argument("-M", "--max", dest="max", type=int, default=None)
    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--strategy", dest="strategy", type=str, default="hypophet")
    parser.add_argument("-p", dest="part", type=int, default=None)
    parser.add_argument("-s", dest="save_name", type=str, default="05.segment")
    args = parser.parse_args()
    return args


def load_config(conf_path: str):
    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_npy(file_path, arr):
    try:
        np.save(file_path, arr)
    except Exception as e:
        print(e)
        return 0
    return 1


def load_npy(file_path):
    try:
        arr = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(e)
        arr = np.array([])
    return arr


def multiprocess_function(f, f_input, processes=4):
    result = []
    pool = mp.Pool(processes=processes)
    with tqdm(total=len(f_input), ncols=100) as pbar:
        for x in pool.imap(f, f_input):
            pbar.update(1)
            result.append(x)
    pool.close()
    pool.join()
    return result


def select_strategy(name):
    match name:
        case "hypophet":
            print("Hypophet strategy selected.")
            strategy = Hypophet()

        case "hypophetval1":
            print("HypophetVal1 strategy selected.")
            strategy = HypophetVal1()

        case "hypophetval2":
            print("HypophetVal2 strategy selected.")
            strategy = HypophetVal2()

        case "hypophetval3":
            print("HypophetVal3 strategy selected.")
            strategy = HypophetVal3()

        case "hypophettest1":
            print("HypophetTest1 strategy selected.")
            strategy = HypophetTest1()

        case "hypophettest2":
            print("HypophetTest2 strategy selected.")
            strategy = HypophetTest2()

        case "hypophettest3":
            print("HypophetTest3 strategy selected.")
            strategy = HypophetTest3()

        case "hypophettest4":
            print("HypophetTest4 strategy selected.")
            strategy = HypophetTest4()

        case "hypophettest5":
            print("HypophetTest5 strategy selected.")
            strategy = HypophetTest5()

        case "hypophettest6":
            print("HypophetTest6 strategy selected.")
            strategy = HypophetTest6()

        case "hypophettest7":
            print("HypophetTest7 strategy selected.")
            strategy = HypophetTest7()

        case "hypophettest8":
            print("HypophetTest8 strategy selected.")
            strategy = HypophetTest8()

        case "hypophetversion2":
            print("HypophetVersion2 strategy selected.")
            strategy = HypophetVersion2()

        case "acumen":
            print("Acumen strategy selected.")
            strategy = Acumen()

        case "acumen2":
            print("Acumen2 strategy selected.")
            strategy = Acumen2()

        case "asan":
            print("Asan strategy selected.")
            strategy = Asan()

        case "moghadam":
            print("Moghadam strategy selected.")
            strategy = Moghadam()

        case "groundtruth":
            print("GroundTruth strategy selected.")
            strategy = GroundTruth()

        case "groundtruth2":
            print("GroundTruth2 strategy selected.")
            strategy = GroundTruth2()

        case _:
            raise Exception("Should select available strategy.")

    return strategy


async def read_vitalfile(src, trks):
    loop = asyncio.get_running_loop()
    desired_vf = partial(vitaldb.VitalFile, track_names=trks)
    vf = await loop.run_in_executor(None, desired_vf, str(src))

    if vf.get_track_names().__len__() < 1:
        raise Exception("There is no track of interest.")

    return vf


def read_best_signal(vf, fs):
    tracks = vf.to_numpy(vf.get_track_names(), 1 / fs, True)
    ts, sig = tracks[:, 0:1], np.float32(tracks[:, 1:])
    # count number of missing values for each track(element-wise)
    is_missing = np.isnan(sig) | (sig == 0)
    selected_track_idx = np.sum(is_missing, axis=0).argmin()
    sig = sig[:, selected_track_idx].reshape(-1, 1)
    return ts, sig


def carry_forward(sig):
    # 만약 배열 내에 값이 한 개라도 존재하면 그것을 토대로 carry-forward는 동작할 것이다.
    # 좌측/우측의 시작이 np.nan 이든, 배열의 중앙이 np.nan 이든 문제 없었을 것이다.
    # 하지만 전체가 np.nan 이면 방법이 없으므로 이에 대한 예외처리 코드를 넣어주어야 한다.
    # (추가)
    # 1초 길이 이내의 데이터가 누락되어 있다면, carry-forward 를 적용한다.
    # 하지만 그 이상의 데이터가 누락되어 있다면, carry-forward 를 중단하고 0으로 채운다.
    # 맥박 1회 내에서의 데이터 누락은 carry-forward 를 하면 복구가 될 수도 있지만, 그 이상은 사실상 복구가 불가하다.
    imputed_sig = sig.copy()

    if np.isnan(imputed_sig).all():
        raise Exception("Whole value of signal is NaN.")

    nan_loc = np.where(np.isnan(imputed_sig))[0]
    if len(nan_loc) != 0:
        nan_area = np.split(nan_loc, np.where(np.diff(nan_loc) != 1)[0] + 1)
        for x in nan_area:
            imputed_sig[x[0] : x[-1] + 1] = imputed_sig[x[0] - 1]

    return imputed_sig


def standardize(arr: np.array):
    """
    Array should be 1-dimensional.
    The function returns standardized array.
    """
    try:
        result = (arr - arr.mean()) / arr.std()
    except Exception as e:
        result = None
        print(e)
    return result


def assess_beat(arr, troughs, approx_zero, thresholds, verbose=False):
    """
    Assessing beat quality using skewness.
    A beat is identified by two troughs.
    """
    sqi = np.ones(len(arr)) * -1
    for i in np.arange(1, len(troughs)):

        beat_size = troughs[i] - troughs[i - 1]
        beat = arr[troughs[i - 1] : troughs[i]]

        scaled_beat = standardize(beat)
        skewness = skew(scaled_beat)

        diff_val = np.diff(scaled_beat)
        isflat = (diff_val > approx_zero[0]) & (diff_val < approx_zero[1])
        flatness = isflat.sum() / len(isflat)

        # y축반전 ABP 신호에 대한 해결방법: beat 를 반으로 나누어 양쪽의 표준편차를 이용해 y축 반전이나 이상한 신호를 제거한다.
        # 정상적인 inv_index 면 양수가 나와야 하며 그 차이가 커야 한다. 만약 양수더라도 차이가 미비하면 안된다.
        l_beat = scaled_beat[: int(beat_size / 2)]
        r_beat = scaled_beat[int(beat_size / 2) :]
        shape_integrity = np.std(l_beat) - np.std(r_beat)

        is_outlier = (
            (beat_size > thresholds[0])
            | np.any(beat > thresholds[1])
            | np.any(beat < thresholds[2])
            | (skewness < thresholds[3])
            | (flatness > thresholds[4])
            | (shape_integrity < thresholds[5])
            | np.isnan(skewness)
            | np.isnan(flatness)
        )
        result = (skewness + flatness) if not is_outlier else -1

        if verbose:
            print(
                (beat_size > thresholds[0]),
                np.any(beat > thresholds[1]),
                np.any(beat < thresholds[2]),
                (skewness < thresholds[3]),
                (flatness > thresholds[4]),
                (shape_integrity < thresholds[5]),
                np.isnan(skewness),
                np.isnan(flatness),
            )
            print(skewness, flatness, result)

        sqi[troughs[i - 1] : troughs[i]] = result

    score = 1 - ((sqi - sqi.min()) / (sqi.max() - sqi.min()))  # 0~1 사이 값으로 표준화
    if sum(np.isnan(score)) == len(score):
        score = np.where(np.isnan(score), 1, score)

    # Updated 25.12.30: 이건 segment 수준(05_load_data.py)에서만 적용한다.
    #   새그먼트의 양쪽 끝은 절단면 때문에 품질지표가 1로 뜬다.
    #   이 부분은 0으로 바꾸어주자.
    # if not np.all(score == 1):
    #     score_section = np.split(score, np.where(np.diff(score, prepend=1) != 0)[0])
    #     score_section[0] = np.zeros_like(score_section[0])
    #     score_section[-1] = np.zeros_like(score_section[-1])
    #     score = np.hstack(score_section)

    return 1 - score  # updated at 2026.10.09.


def get_peak_trough(arr, config):
    custom_fp = partial(
        find_peaks,
        distance=config["dist"],
        prominence=config["prom"],
        width=config["width"],
        wlen=config["wlen"],
    )

    peaks = custom_fp(arr.reshape(-1), height=config["peak_height"])[0]
    troughs = custom_fp(-1 * arr.reshape(-1), height=config["trough_height"])[0]
    return peaks, troughs


def moving_average(arr, N, mode):
    return np.convolve(arr, np.ones(N) / N, mode)


def plot_sqi(sample, config):
    """
    Draw a signal with it's informations.
    - Blue line(up): raw signal
    - Red points: peak & trough
    - Orange line: signal quality index
    - Red dotted line: threshold for sqi
    - Blue line(down): qualified signal
    - Magenta line: mean arterial pressure score
    """
    custom_fp = partial(
        find_peaks,
        distance=config["dist"],
        prominence=config["prom"],
        width=config["width"],
        wlen=config["wlen"],
    )
    peaks = custom_fp(sample.reshape(-1), height=config["peak_height"])[0]
    troughs = custom_fp(-1 * sample.reshape(-1), height=config["trough_height"])[0]

    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    axs[0].plot(sample)
    axs[0].plot(peaks, sample[peaks], "^", color="red")
    axs[0].plot(troughs, sample[troughs], "o", color="red")

    # beat-to-beat
    sqi, map_idx = np.ones(len(sample)), np.zeros(len(sample))
    for i in np.arange(1, len(troughs)):
        beat = sample[troughs[i - 1] : troughs[i]]
        scaled_beat = standardize(sample[troughs[i - 1] : troughs[i]])
        sqi[troughs[i - 1] : troughs[i]] = np.exp(-skew(scaled_beat))
        map_idx[troughs[i - 1] : troughs[i]] = 65.0 / np.mean(beat)

    twin_axs0 = axs[0].twinx()  # Create a twin y-axis
    twin_axs0.plot(sqi, color="orange")
    # Plot test_sqi1 on the secondary y-axis
    twin_axs0.axhline(config["threshold"], color="red", linestyle="--")

    masked_sample = np.where(sqi < config["threshold"], sample.reshape(-1), np.nan)
    map_idx = np.where(np.isnan(masked_sample), np.nan, map_idx)
    axs[1].plot(masked_sample)
    twin_axs1 = axs[1].twinx()  # Create a twin y-axis
    twin_axs1.plot(map_idx, color="magenta")
    plt.show()
    return masked_sample, peaks, troughs, sqi, map_idx


def calculate_meanbp(arr, troughs):
    """
    Calculate Mean BP per beat.
    A beat is identified by two troughs.
    """
    meanbp = np.zeros(len(arr))
    for i in np.arange(1, len(troughs)):
        beat = arr[troughs[i - 1] : troughs[i]]
        meanbp[troughs[i - 1] : troughs[i]] = np.mean(beat)

    return meanbp


def process_meanbp(sig, config):
    try:
        _, troughs = get_peak_trough(sig, config["detection"])
        meanbp = np.zeros(len(sig))
        for i in np.arange(1, len(troughs)):
            beat = sig[troughs[i - 1] : troughs[i]]
            meanbp[troughs[i - 1] : troughs[i]] = np.mean(beat)
    except Exception as e:
        print(f"Calculating Error: {e}")
        meanbp = None

    return meanbp
