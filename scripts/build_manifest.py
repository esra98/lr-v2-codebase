#!/usr/bin/env python
"""Write the train, validation and test manifests of a dataset.

The split was fixed when the dataset was built, because only training clips
carry Partition-Time Masking. This script reads it from the clip table
written by scripts/build_dataset.py.

Usage:
    python scripts/build_manifest.py --dataset data/tr-lrs-clean-ptm
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avsr.data.manifest import kept_rows, read_clip_table, write_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset folder made by build_dataset.py")
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument("--max-frames", type=int, default=600, help="Leave out longer clips")
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    table_path = dataset + ".clips.tsv"
    if not os.path.isfile(table_path):
        sys.exit(f"Clip table not found: {table_path}. Build the dataset first.")
    table = read_clip_table(table_path)
    print(f"dataset: {dataset}")
    empty = []
    for split in ("train", "val", "test"):
        rows = [r for r in kept_rows(table, split) if r.num_frames <= args.max_frames]
        absent = [r.utt_id for r in rows if not os.path.isfile(os.path.join(dataset, r.session, r.clip_id + ".mp4"))]
        if absent:
            sys.exit(f"{len(absent)} {split} clips of the table are missing from {dataset}, for example {absent[0]}")
        hours = sum(r.num_frames for r in rows) / 25 / 3600
        write_manifest(os.path.join(args.out, f"{split}.tsv"), rows)
        print(f"{split:5s}: {len(rows):6d} utterances, {hours:6.2f} h, {len({r.session for r in rows})} sessions")
        if not rows:
            empty.append(split)
    if empty:
        sys.exit(f"Empty split: {', '.join(empty)}")


if __name__ == "__main__":
    main()
