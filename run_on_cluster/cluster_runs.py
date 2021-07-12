# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 14:52:21
# @Last Modified: 2021-07-05 15:35:37
# ------------------------------------------------------------------------------ #
# This script should be called with the submit.sh script, it takes the ids and
# maps them to our model combinations

import argparse
import logging
import os
from multiprocessing import Pool
import itertools

log = logging.getLogger("ClusterRunner")

parser = argparse.ArgumentParser(description="Run variant script")
parser.add_argument(
    "-i", "--id", type=int, help="ID", required=True,
)

args = parser.parse_args()
args.id = args.id - 1
log.info(f"ID: {args.id}")


""" Get possible different combinations
"""
mapping = []


possible_params_0 = ["beta","binom","dirichlet"]
possible_params_1 = ["SIR","kernelized_spread"]

for i in possible_params_0:
    for j in possible_params_1:
        mapping.append([i,j])


def exec(likelihood, spread_method):
    """
    Executes python script
    """
    os.chdir("../")
    os.system(
        f"python run_model.py -l {likelihood} -s {spread_method}"
    )


exec(*mapping[args.id])
