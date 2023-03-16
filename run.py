import argparse
import subprocess
import torch
from collections import defaultdict

"""
This is a small wrapper to repeat training multiple times (based on the OGB instructions)
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="dgl",
        choices=["pyg", "dgl"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        choices=["ogbg-molhiv", "ogbg-molpcba"]
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10
    ) 
    return parser.parse_args()

def run(cmd):
    print(f"%%%%%%%%%%%%%%%%%%%%%")
    print("Executing command: ")
    print(cmd)
    popen = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_lines in iter(popen.stdout.readline, ""):
        print(stdout_lines)
    popen.stdout.close()

if __name__ == '__main__':
    args = parse_args()
    base_cmd = f"python  {args.backend}_distributed.py --batch_size {args.batch_size} --dataset {args.dataset} --epoch {args.epoch}"
    for world_size in [1, 2, 4, 8]:
        cmd = base_cmd + f" --world_size {world_size}"
        run(cmd)