import argparse
import os
import subprocess
import json
import sys

from concurrent import futures


if __name__ == "__main__":
    print("Have you updated the paths to your data? (y/n)")
    resp = input()
    if resp.lower() != "y":
        print("Please modify scripts/dataset_extract.py")
        sys.exit()
    print("Comment this out in scripts/dataset_extract.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default=None, nargs="*")
    parser.add_argument("--root", default="/path/to/custom_videos")
    parser.add_argument("--specs", default="custom_specs.json")
    args = parser.parse_args()

    vid_root = os.path.join(args.root, "videos")
    out_root = os.path.join(args.root, "PNGImages")
    with open(args.specs, "r") as f:
        time_specs = json.load(f)

    if args.seqs is None:
        args.seqs = time_specs.keys()

    with futures.ProcessPoolExecutor(max_workers=4) as ex:
        for seq in args.seqs:
            if seq not in time_specs:
                print(f"{seq} not specified")
                continue

            vid_path = "{}/{}.mp4".format(vid_root, seq)
            if not os.path.isfile(vid_path):
                print(vid_path, "does not exist!")
                continue

            spec = time_specs[seq]
            cmd = "python extract_frames.py {} {} --start {} --end {} --fps {}".format(
                vid_path, out_root, spec["start"], spec["end"], spec["fps"]
            )
            print(cmd)
            ex.submit(subprocess.call, cmd, shell=True)
