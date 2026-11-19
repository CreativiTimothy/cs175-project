import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--all-active", action="store_true")
    parser.add_argument("--symbols-file")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--end", required=True)
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    for name, settings in config["data"]["timeframes"].items():
        command = [
            sys.executable, "-m", "src.data.bulk_alpaca_downloader",
            "--timeframe", settings["alpaca_timeframe"],
            "--start", settings["download_start"],
            "--end", args.end,
            "--output", settings["path"],
            "--chunk-size", str(args.chunk_size),
        ]
        if args.symbols_file:
            command += ["--symbols-file", args.symbols_file]
        else:
            command += ["--all-active"]
            if args.max_symbols:
                command += ["--max-symbols", str(args.max_symbols)]
        print("\n", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
