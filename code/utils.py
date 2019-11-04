"""Contains miscellaneous utility functions for Swift data analysis."""

import multiprocessing as MP


def _worker_trajectories(params):
    arr, user_idx, user_id, session_threshold = params



def mk_user_trajectories(df, session_threshold=5 * 60):
    """Takes in a dataframe in HLR format and returns user sessions times and
    sizes.

     :session_threshold: is the minimum time difference between two sessions.
    """





