#!/usr/bin/env python
import glob
import re
import os

import click
import pandas as pd
import numpy as np
import multiprocessing as MP

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# from dataworkspaces.lineage import LineageBuilder


ARGS_REGEX = {
    "hlwt": (re.compile(r"hlwt-([^,]*)"), float),
    "l2wt": (re.compile(r"l2wt-([^,]*)"), float),
    "epochs": (re.compile(r"epochs-([0-9]*)"), int),
    "method": (re.compile(r"method-([^,]*)"), str),
    "shuffle": (re.compile(r"shuffle-([0-9]*)"), int),
    "training_fraction": (re.compile(r"training_fraction-([^,]*)"), float),
    "dataset": (re.compile(r",([^,]*)\.preds"), str),
}


def parse_args(file_name):
    """Extract dictionary of arguments from a given file_name."""
    args = {}
    for arg_name, (arg_regex, arg_type) in ARGS_REGEX.items():
        match_obj = arg_regex.search(file_name)
        if match_obj:
            args[arg_name] = arg_type(match_obj[1])
        else:
            args[arg_name] = None
    return args


def _analysis_worker(op_file):
    preds = pd.read_csv(op_file, sep="\t")
    args = parse_args(op_file)

    args["MAE"] = np.mean(np.abs(preds["p"] - preds["pp"]))
    args["AUC"] = roc_auc_score(preds["p"], preds["pp"])
    args["COR_p"] = spearmanr(preds["p"], preds["pp"])[0]
    args["COR_h"] = spearmanr(preds["h"], preds["hh"])[0]

    return args


@click.command()
@click.argument("results_dir", type=click.Path())
@click.argument("output_csv", type=click.Path())
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Run in single threaded mode for debugging.",
)
def run(results_dir, output_csv, debug):
    """Read all *.detailed files from RESULTS_DIR, calculate the metrics, and
    save output to OUTPUT_CSV."""
    op_files = glob.glob(os.path.join(results_dir, "*.preds"))

    # builder = (
    #     LineageBuilder()
    #     .as_script_step()
    #     .with_parameters({
    #         'results_dir': results_dir,
    #     })
    #     .with_input_paths(op_files)
    # )

    # with builder.eval() as lineage:
    if True:
        if debug:
            data = [_analysis_worker(op_file) for op_file in op_files]
        else:
            with MP.Pool() as pool:
                data = pool.map(_analysis_worker, op_files)

        pd.DataFrame(data).to_csv(output_csv, index=False)
        # lineage.add_output_path(output_csv)

    print("Done.")


if __name__ == "__main__":
    run()
