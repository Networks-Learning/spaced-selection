#!/usr/bin/env python

import click
import os
import numpy as np

SLURM_OUTPUT_DIR = "slurm-output"


@click.command()
@click.argument("input_csv")
@click.argument("output_dir")
@click.option(
    "--slurm-output-dir",
    help="Where to save the output",
    default=SLURM_OUTPUT_DIR,
    show_default=True,
)
@click.option("--dry/--no-dry", help="Dry run.", default=True, show_default=True)
@click.option("--epochs", help="Epochs.", default=500, show_default=True)
@click.option(
    "--mem",
    help="How much memory will each job need (MB).",
    default=10000,
    show_default=True,
)
@click.option("--timeout", help="Minutes to timeout.", default=120)
@click.option("--shuffle", help="Seed to shuffle training/testing using.", default=41)
@click.option("--l-rate", help="Initial learning rate.", default=0.01)
def run(
    input_csv, output_dir, slurm_output_dir, epochs, dry, mem, timeout, shuffle, l_rate
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(slurm_output_dir, exist_ok=True)

    input_csv_abs = os.path.abspath(input_csv)

    for model in ["hlr", "power"]:
        for h_reg in np.logspace(-9, 2, 15):
            for l2wt in np.logspace(-9, 2, 15):
                stdout_file = f"{slurm_output_dir}/model,{model}-h_reg,{h_reg:.9f}-l2wt,{l2wt:.9f}-%j.out"
                cmd = (
                    f'sbatch -c 1 --mem={mem} -o "{stdout_file}" '
                    + f"--time={timeout} "
                    + f"./code/slurm/grid_search_job.sh "
                    + f"{input_csv_abs} {model} {epochs} "
                    + f"{shuffle} {h_reg:.9f} {l2wt:.9f} "
                    + f'"{output_dir}" {l_rate:.9f}'
                )
                print(cmd)

                if not dry:
                    print("Running ...")
                    os.system(cmd)


if __name__ == "__main__":
    run()
