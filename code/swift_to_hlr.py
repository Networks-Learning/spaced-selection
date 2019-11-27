#!/usr/bin/env python
import glob
import os
import sys
import multiprocessing as MP
import datetime as D

import click
import pandas as pd
import numpy as np

# from dataworkspaces.lineage import LineageBuilder


def _now(raw=False):
    """Return the time now in red color."""
    templ = "\x1b[31m[{}]\x1b[0m" if not raw else "{}"
    return templ.format(D.datetime.now().isoformat(timespec="seconds"))


# Specifying the format is ~ 10x faster.
swift_datetime_format = "%a %b %d %Y %H:%M:%S GMT+0000 (%Z)"


def _worker_read_csv(filename):
    df_tmp = pd.read_csv(filename)
    df_tmp.datecreated = pd.to_datetime(
        df_tmp.datecreated, format=swift_datetime_format
    )
    df_tmp.dateinstallation = pd.to_datetime(
        df_tmp.dateinstallation, format=swift_datetime_format
    )
    return df_tmp


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_hlr_csv", type=click.Path())
@click.argument("output_sim_csv", type=click.Path())
@click.option(
    "--verbose/--no-verbose", default=True, help="Verbose output.", show_default=True
)
@click.option(
    "--force/--no-force", default=False, help="Overwrite output.", show_default=True
)
@click.option(
    "--min-count",
    default=1,
    help="Minimum number of times a user must have practiced a question to include it for training/prediction.",
    show_default=True,
)
@click.option(
    "--results-dir",
    default="results",
    help="The results folder for Lineage.",
    show_default=True,
)
def run(
    input_dir, output_hlr_csv, output_sim_csv, verbose, force, min_count, results_dir
):
    """Convert CSV files in INPUT_DIR from Swift.ch format to Duolingo's HLR
    format and save in OUTPUT_HLR_CSV, as well as to extract each attempe and
    save in OUTPUT_SIM_CSV."""

    # builder = (
    #     LineageBuilder()
    #     .as_script_step()
    #     .with_parameters({
    #         'force': force,
    #         'min_count': min_count,
    #     })
    #     .as_results_step(os.path.join(results_dir))
    # )

    if (os.path.exists(output_hlr_csv) or os.path.exists(output_sim_csv)) and not force:
        print(
            _now(),
            "{} or {} exists and --force not supplied.".format(
                output_hlr_csv, output_sim_csv
            ),
        )
        sys.exit(-1)

    data_files = glob.glob(os.path.join(input_dir, "stats_2019????.csv"))

    # builder = builder.with_input_paths(data_files)

    if verbose:
        print(_now(), "Total files found = ", len(data_files))
        print(_now(), "Starting reading ...")

    # with builder.eval() as lineage:
    if True:
        with MP.Pool() as pool:
            df = pd.concat(pool.map(_worker_read_csv, data_files))

        if verbose:
            all_data_size = df.shape[0]
            print(_now(), "Reading finished, read {} rows.".format(all_data_size))

        df.rename(
            columns={
                "correct": "p_recall",
                "datecreated": "timestamp",
                "user": "user_id",
                "question": "lexeme_id",
                "count": "history_seen",
                "language": "learning_language",
            },
            inplace=True,
        )
        df["ui_language"] = "de"
        df["lexeme_string"] = df["lexeme_id"]
        df["session_seen"] = 1
        df["session_correct"] = df["p_recall"]
        # The training timestamp should be in days.
        df["timestamp"] = df["timestamp"].values.astype(np.int64) // 10 ** 9

        # Determine the time delta between attempts
        df.sort_values(by=["user_id", "lexeme_id", "timestamp"], inplace=True)
        df["delta"] = df["timestamp"].diff()

        # Find the places where the user_id, lexeme_id changes
        change_loc = (df.lexeme_id != df.lexeme_id.shift()) | (
            df.user_id != df.user_id.shift()
        )
        df.loc[change_loc, "delta"] = None

        # Populate the history of correct answers
        history_correct = df.p_recall.cumsum()
        history_correct_correction = history_correct.copy()
        history_correct_correction[:] = np.nan
        history_correct_correction[df.delta.isnull()] = (
            history_correct[df.delta.isnull()] - df.p_recall[df.delta.isnull()]
        )
        history_correct -= history_correct_correction.fillna(method="ffill")
        df["history_correct"] = history_correct.astype(int)

        if verbose:
            print(
                _now(),
                "About {:.2f}% of the data has been answered correctly more "
                "than the times it has been seen.".format(
                    100
                    * df[df["history_seen"] < df["history_correct"]].shape[0]
                    / df.shape[0]
                ),
            )

        # Drop part of the data where history_seen > history_correct
        df = df[df["history_seen"] >= df["history_correct"]]

        # Copy all the sessions
        df_items = df[["timestamp", "lexeme_id", "user_id", "p_recall"]]

        if verbose:
            print(_now(), "Saving all sessions ...")

        df_items.to_csv(output_sim_csv, index=False)
        # Drop the first attempts to various user_id, question pairs

        df.dropna(inplace=True)

        if verbose:
            print(
                _now(),
                "Total number of usable attempts = {}/{} = {:.2f}%".format(
                    df.shape[0], all_data_size, df.shape[0] / all_data_size * 100.0
                ),
            )

        if verbose:
            print(_now(), "Pruning to at least {} attempts ...".format(min_count))

        df_grouped = df.groupby(["user_id", "lexeme_id"]).timestamp.size()
        df_filtered = df_grouped[df_grouped >= min_count]
        idx_filtered = sorted(
            {
                x[0] + "/" + x[1]
                for x in df_filtered.reset_index()[["user_id", "lexeme_id"]].values
            }
        )
        df_user_lexeme = (df.user_id + "/" + df.lexeme_id).values.tolist()

        if verbose:
            print(
                _now(),
                "Keeping {} / {} (user, question) pairs.".format(
                    len(idx_filtered), df_grouped.shape[0]
                ),
            )

        # Using merging of sorted arrays to determine which part of the
        # dataframe to keep.
        # The assumption is that grouping will sort the index.
        select_idx = [False] * len(df_user_lexeme)
        i, j = 0, 0
        while i < len(df_user_lexeme) and j < len(idx_filtered):
            if df_user_lexeme[i] == idx_filtered[j]:
                select_idx[i] = True
                i += 1
            elif df_user_lexeme[i] < idx_filtered[j]:
                i += 1
            elif df_user_lexeme[i] > idx_filtered[j]:
                j += 1

        df = df[select_idx]

        if verbose:
            print(_now(), "Keeping {} rows ...".format(df.shape[0]))

        # lineage.add_output_path(output_hlr_csv)
        df.to_csv(output_hlr_csv)

        # lineage.write_results({
        #     'size_filtered': len(idx_filtered),
        #     'size_raw': df_grouped.shape[0],
        # })

        if verbose:
            print(_now(), "Done.")


if __name__ == "__main__":
    run()
