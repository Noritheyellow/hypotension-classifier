import argparse
import numpy as np
from src.module.imputation import process_impute
from src.module.noise_filtering import process_filter, process_smooth
from src.module.utils import process_meanbp
from src.module.signal_quality_assessment import process_assess


def preprocess(event_idx, caseid, sig, ts, config):
    """
    Args:
    - `event_idx`: shape=(N, 2), where N is number of events, axis=1 is (start, end).
    - `caseid`: str, for example '0001', '0014', etc.
    - `sig`: raw signal. type of np.array.
    - `ts`: corresponding datetime.
    """
    cid = np.array([caseid] * len(np.arange(*event_idx)))
    seg = sig[np.arange(*event_idx)]
    if sum(np.isnan(seg)) == len(seg):
        return None
    try:
        proc_seg = process_impute(seg)
        proc_seg = process_filter(proc_seg, config)
        proc_seg = process_smooth(proc_seg, config)
        seg_ts = ts[np.arange(*event_idx)]
        seg_mbp = process_meanbp(proc_seg, config)
        sqi = process_assess(proc_seg, config)
        seg_ds = np.vstack([cid, seg_ts, proc_seg, seg_mbp, sqi]).T
        seg_ds = np.expand_dims(seg_ds, 0)
    except Exception as e:
        print(e)
        seg_ds = None
    return seg_ds


def prepare(segment, config, i):
    static_len = config["fs"] * config["static_len"]
    cid, seg_ts, proc_seg, seg_mbp, sqi = segment.T
    seg_ts = seg_ts.astype(object)
    proc_seg = proc_seg.astype("float32")
    seg_mbp = seg_mbp.astype("float32")
    sqi = sqi.astype("float32")

    # 새그먼트 내에서 NaN 인 부분을 제외한 부분새그먼트의 start, end
    masked_sig = np.where(sqi > config["sqi"]["threshold"], np.nan, proc_seg)
    if np.isnan(masked_sig).sum() != 0:
        starts = np.where(np.diff(~np.isnan(masked_sig), prepend=0) > 0)[0]
        ends = np.where(np.diff(~np.isnan(masked_sig), prepend=0) < 0)[0]
        seg_idx = np.array(tuple(zip(starts, ends)))
    else:
        seg_idx = np.array((0, len(masked_sig)))[np.newaxis, :]

    # 새그먼트 내 각 부분새그먼트의 길이 계산
    try:
        part_len = np.diff(seg_idx, axis=1)
        start, end = seg_idx[np.where(part_len >= static_len)[0]].reshape(-1)
        cid = np.array([[cid[0]] * static_len])
        seg_ts = np.expand_dims(seg_ts[start:end][:static_len], 0)
        segment = np.expand_dims(masked_sig[start:end][:static_len], 0)
        result = np.concatenate([cid, seg_ts, segment], axis=0).T[np.newaxis, :, :]
    except ValueError as e:
        # print(f"{cid[0]} {i}th segment: {e}({part_len[0][0]} >= {static_len})")
        return None

    return result


if __name__ == "__main__":
    """
    Description:
        prepare_dataset.py 는 `src` 에 위치한 부분 신호 `*[hypo|normo]*sqi.npy` 를 읽는다.
        읽은 부분 신호로부터 `config['threshold']` 에 따라 일부를 np.nan 으로 마스킹한다.
        마스킹한 신호에서 np.nan 이 아닌 지점들만 추출한다.
        만약 하나의 부분 신호에서 시작/종료지점 이내에 np.nan 이 존재한다면 해당 지점을 배제하고 부분 신호는 복수의 새그먼트로 분할된다.
        따라서 하나의 부분 신호는 1개 이상의 새그먼트를 가질 수 있다.
        추출된 새그먼트 중에서 `config['static_len']` 초 이상의 길이를 가진 신호 중 첫번째가 데이터셋으로 선택된다.
        저혈압은 positive 이며 '1' 로 라벨링되고, 정상혈압은 negtive 로 '0' 으로 라벨링된다.
        최종적으로 이들은 `dest` 에 `pos_dataset.npy`, `pos_label.npy`, `neg_dataset.npy`, `neg_label.npy` 로 저장된다.

    Arguments:
        -c conf : configure yaml 파일의 경로
        -s src  : 전체 혹은 부분 신호를 읽을 경로(전체 신호는 `*.vital`, 부분 신호는 `*.npy` 여야 한다. 자세한 건 각 함수에서 확인.)
        -d dest : 결과가 저장될 경로

    Command line:
        1) python src/v02/pipeline/prepare_dataset.py -c config/conf_v02.yaml -s dataset/v02/part_preprocess -d dataset/v02/dataset
    - python src/v02/pipeline/prepare_dataset.py -c config/conf_v02.yaml -s /Volumes/My\ Book/DataWarehouse/SNU-Hypo10Pred/v02/part_preprocessv021 -d /Volumes/My\ Book/DataWarehouse/SNU-Hypo10Pred/v02/dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", dest="method", type=int, help="select method: [1, 2]"
    )
    parser.add_argument("-c", "--conf", dest="conf", type=str)
    parser.add_argument("-s", "--src", dest="src", type=str)
    parser.add_argument("-d", "--dest", dest="dest", type=str)
    args = vars(parser.parse_args())

    main(**args)
