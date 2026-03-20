import argparse
import datetime
from pathlib import Path
import asyncio
import yaml
import module.resampling as resampling

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf", dest="conf", type=str)
parser.add_argument("-m", "--min", dest="min", type=int, default=None)
parser.add_argument("-M", "--max", dest="max", type=int, default=None)
args = parser.parse_args()

# Usage: python src/extract_data.py -c config/conf.yaml -m 0 -M 10
if __name__ == "__main__":
    conf_path = Path(args.conf)
    with open(conf_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    src_path = Path(config["src_path"]).expanduser()
    save_path = Path(config["data_path"]).expanduser() / "01.raw"
    file_path = sorted(list(src_path.rglob("*.vital")))[args.min : args.max]

    print(f"Configuration")
    print(f"- source path: {src_path}")
    print(f"- save path: {save_path}")
    print(f"- track names: {config['track_names']}")
    print(f"- frequency: {config['fs']}Hz")
    print(f"- number of files: {len(file_path)}")

    start_time = datetime.datetime.now()
    result = asyncio.run(
        resampling.async_process(
            file_path, config["fs"], config["track_names"], save_path
        )
    )
    end_time = datetime.datetime.now()
    print(f"Elapsed time: {end_time - start_time}")
