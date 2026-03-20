import os
from pathlib import Path
from functools import partial
import h5py
import numpy as np
from scipy.signal import resample_poly
import base64
from functools import reduce
import xml.etree.ElementTree as ET
from tqdm import tqdm
from src.module.utils import parse_args, load_config, multiprocess_function


def set_environment(args, config):
    src_path = Path(config["src_path"]).expanduser()
    save_path = Path(config["output_path"]).expanduser()
    env = {
        "src_path": src_path,
        "wave_path": src_path
        / "epic_wave_3_v2"
        / "UCI_deidentified_part4_EPIC_11_28"
        / "Waveforms",
        "save_path": save_path / "01.raw",
    }
    return env


def set_bit(v, index, x):
    """
    Set the index:th bit of v to 1 if x is truthy,
    else to 0, and return the new value.
    """
    mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
    v &= ~mask  # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask  # If x was True, set the bit indicated by the mask.
    return v  # Return the result, we're done.


def cond_for_file(file_path):
    # 한 Epic_wave 디렉터리 하위의 모든 생체신호 파일 경로를 가져온다.
    cond1 = file_path.is_file()
    cond2 = file_path.name[0] != "."
    cond3 = file_path.name.split("-")[0][-2:] == "IP"
    return cond1 and cond2 and cond3


def read_measurement(files):
    measurements = []
    for f in files:
        tree = ET.parse(f)
        root = tree.getroot()
        measurements.append(root.findall(".//mg"))
    measurements = reduce(lambda x1, x2: x1 + x2, measurements)
    return measurements


def read_attribute(measurement, sig_name):
    waves, offsets, gains, hzs, points = [], [], [], [], []
    for mg in measurement:
        if mg.get("name") == sig_name:
            for m in mg:
                if m.attrib["name"] == "Offset":
                    offsets.append(int(m.text))
                elif m.attrib["name"] == "Gain":
                    if mg.get("name") == "GE_ART":
                        gains.append(0.25)
                    elif mg.get("name") == "INVP1":
                        gains.append(0.01)
                    else:
                        gains.append(float(m.text))

                elif m.attrib["name"] == "Wave":
                    waves.append(m.text)
                elif m.attrib["name"] == "Hz":
                    hzs.append(int(m.text))
                elif m.attrib["name"] == "Points":
                    points.append(int(m.text))

    return waves, offsets, gains, hzs, points


def decode_wave(wave, offset, gain):
    binwaves = []
    for j, wave_enc in enumerate(wave):
        binwave = []
        wave_dec = base64.b64decode(wave_enc)
        for i in range(0, len(wave_dec) - 1, 2):
            t = (wave_dec[i]) + wave_dec[i + 1] * 256
            t = set_bit(t, 15, 0) + (-32768) * (t >> 15)
            t = t * gain[j] + offset[j]
            binwave.append(t)

        binwaves.append(binwave)
    binwaves = np.array(binwaves)
    return binwaves


def run(log_id, wave_file):
    """
    30분으로 wave 파일들이 기록되며, 각 파일은 서로 다른 환자의 것일 수도 있고 동일 환자의 것일 수도 있다.
    만약 동일환자의 wave 파일이 여러개라면 전부 읽어서, 전체 신호 데이터 추출.
    """
    try:
        matched_files = sorted(filter(lambda f: log_id in f.name, wave_file))
        measurement = read_measurement(matched_files)
        wave, offset, gain, hz, point = read_attribute(measurement, "GE_ART")
        binwave = decode_wave(wave, offset, gain)
        sig = binwave.reshape(-1).astype(np.float32)
        resamp_sig = resample_poly(sig, up=5, down=9)  # 180 hz -> 100 hz
        hz = [100]

        if (resamp_sig.mean() < 25.5) or (resamp_sig.std() > 160):
            return None

    except Exception as e:
        print(f"LOG_ID {log_id}: {e}")
        return None

    return log_id, hz[0], point[0], offset[0], gain[0], resamp_sig


def save_h5(env, data):
    log_id, hz, point, offset, gain, sig = data
    save_path = env["save_path"] / log_id
    os.makedirs(save_path, exist_ok=True)
    with h5py.File(save_path / f"{log_id}_{hz}fs.h5", "w") as f:
        f.attrs["id"] = log_id
        f.attrs["fs"] = hz
        f.attrs["point"] = point
        f.attrs["offset"] = offset
        f.attrs["gain"] = gain
        f.create_dataset("signal", data=sig, compression="gzip")


def main():
    args = parse_args()
    config = load_config(args.conf)
    env = set_environment(args, config)
    wave_files = sorted(filter(cond_for_file, env["wave_path"].rglob("*")))
    caseid_list = sorted(
        set(list(map(lambda w: w.name.split("-")[0][:-2], wave_files)))
    )
    print("Total caseid: ", len(caseid_list))
    result = multiprocess_function(
        partial(run, wave_file=wave_files),
        caseid_list[args.min : args.max],
        processes=args.process,
    )
    result = list(filter(lambda x: x is not None, result))
    for x in tqdm(result, ncols=75):
        save_h5(env, x)


"""
* Usage: python src/01_read_mover.py [options...]
*   -c, --conf              Set path for `conf.yaml`.
*   -m, --min               Set start index of wave files.
*   -M, --max               Set end index of wave files.
*   -p, --process           Set number of processor while multiprocessing.
"""
if __name__ == "__main__":
    main()
