#!/usr/bin/env python
import os
import sys

import click
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict
from itertools import cycle, islice
from typing import Dict, Optional, Any, List
import multiprocessing as MP

# Memory model types
POWER = 'POWER'
HLR = 'HLR'
REPLAY = 'REPLAY'

# Item Selector types
ROUND_ROBIN = 'ROUND_ROBIN'
SPACED_SELECTION = 'SPACED_SELECTION'
RANDOMIZED = 'RANDOMIZED'
REPLAY_SELECTOR = 'REPLAY_SELECTOR'
SPACED_SELECTION_DYN = 'SPACED_SELECTION_DYN'
SPACED_RANKING = 'SPACED_RANKING'

# various constraints on parameters and outputs
MIN_HALF_LIFE = 1.0 / (24 * 60)     # 1 minutes (in days)
MAX_HALF_LIFE = 274.                # 9 months  (in days)
FIVE_MINUTES = (5 * 60)             # 5 minutes (in secs)
SEC_IN_DAY = 24 * 60 * 60


def pclip(p: float) -> float:
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def hclip(h: float) -> float:
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def read_difficulty(filename: str, df_attempts: pd.DataFrame,
                    kind: str, seed: int) -> Dict[str, Any]:
    """Reads a Memory Model parameters from the given file.
    The kind parameter controls whether a 'power'-law memory model is read or
    an 'hlr' model.
    """

    if kind == POWER:
        reserved = ['A', 'B', 'right', 'wrong', 'bias']
    elif kind == HLR:
        reserved = ['right', 'wrong', 'bias']
    else:
        raise ValueError('Unknown model type: {}'.format(kind))

    difficulty: Dict[str, float] = {}
    opts = {
        'difficulty': difficulty,
        'kind': kind,
        'seed': seed,
    }

    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split()
            if key in reserved:
                opts[key] = float(value)
                reserved.remove(key)
            else:
                difficulty[key] = float(value)

    opts['m_0'] = (
        df_attempts
        .sort_values('timestamp')
        .groupby(['lexeme_id', 'user_id'])
        .first()
        .groupby('lexeme_id')
        .p_recall
        .mean()
    )
    return opts


def change_memory_model_seed(memory_model_opts: Dict[str, Any], new_seed: int):
    cp_mem_model_opts = memory_model_opts.copy()
    cp_mem_model_opts['seed'] = new_seed
    return cp_mem_model_opts


def mk_sessions(df_user, threshold=FIVE_MINUTES, with_items=False,
                with_datetime=False, as_df=False):
    """Extracts sessions from sequence of item attempts for one user."""
    session_starts, session_lens, session_items = [], [], []
    last_time = -float('inf')

    assert df_user.user_id.nunique() == 1, "More than one user in the dataframe."
    df = df_user.sort_values(by='timestamp')

    for _row_idx, row in df.iterrows():
        time = row.timestamp
        item = row.lexeme_id
        recall = row.p_recall

        if time - last_time > threshold:
            session_starts.append(time)
            session_lens.append(1)
            session_items.append({item: recall})
        else:
            # If more than one attempt was made in the same session,
            # count only the first attempt for the recall.
            # However, they do show that the user was willing to learn
            # these many items in a single session, we could have chosen
            # the items more judiciously
            session_lens[-1] += 1

            if item not in session_items[-1]:
                session_items[-1][item] = recall

        last_time = time

    ret = {
        'session_start': session_starts,
        'session_len': session_lens,
    }

    if with_datetime:
        ret['session_start_datetime'] = [datetime.datetime.fromtimestamp(x)
                                         for x in session_starts]

    if with_items:
        ret['session_items'] = session_items

    if not as_df:
        return ret
    else:
        return pd.DataFrame(ret)


def read_sessions(filename: str, threshold: float = FIVE_MINUTES) -> pd.DataFrame:
    """Read CSV file of sessions."""
    return pd.read_csv(filename)


class MemoryModel:
    def __init__(
        self,
        seed: int,
        kind: str,
        difficulty: Dict[str, float],
        m_0: Dict[str, float],
        **params
    ):
        self.RS = np.random.RandomState(seed=seed)
        self.difficulty = difficulty
        self.kind = kind
        self.m_0 = m_0

        if kind == POWER:
            self.A = params['A']
            self.B = params['B']
            self.right = params['right']
            self.wrong = params['wrong']
            self.bias = params['bias']

        elif kind == HLR:
            self.right = params['right']
            self.wrong = params['wrong']
            self.bias = params['bias']

        else:
            raise ValueError('Unknown model type: {}'.format(kind))

        # It is assumed that the item initially is completely unknown,
        # though there may be better ways of initializing them
        never: Optional[int] = None
        self.item_memory: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'right': 0,
            'wrong': 0,
            'n0': np.mean(list(self.difficulty.values())),
            't_last': never
        })

        for item, n0 in self.difficulty.items():
            self.item_memory[item]['n0'] = n0

    def halflife(self, item: str):
        item_state = self.item_memory[item]
        log_halflife = (
            item_state['right'] * self.right +
            item_state['wrong'] * self.wrong +
            item_state['n0'] +
            self.bias
        )

        try:
            h = hclip(2. ** log_halflife)
        except OverflowError:
            h = MAX_HALF_LIFE

        return h

    def get_recall_prob(self, item: str, time: float):
        """Return the probability of correctly recalling the given item
        at the given time."""

        # TODO: Can be made faster by calculating and caching the half-life
        # as soon as a review is done.

        if self.kind == POWER:
            h = self.halflife(item=item)
            item_state = self.item_memory[item]
            if self.have_reviewed(item):
                assert item_state['t_last'] is not None
                dt = (time - item_state['t_last']) / SEC_IN_DAY
                prob = self.A * (1 + self.B * dt) ^ (-1 / h)
            else:
                # If have not seen, then prob of recall = m_i(0)
                prob = self.m_0[item]
            return pclip(prob)

        elif self.kind == HLR:
            h = self.halflife(item=item)
            item_state = self.item_memory[item]
            if self.have_reviewed(item):
                assert item_state['t_last'] is not None
                dt = (time - item_state['t_last']) / SEC_IN_DAY
                prob = 2. ** (-dt / h)
            else:
                # If have not seen, then prob of recall = m_i(0)
                prob = self.m_0[item]
            return pclip(prob)

        else:
            raise ValueError('Unknown model type: {}'.format(self.kind))

    def have_reviewed(self, item):
        """Returns True if the item has been reviewed in the past."""
        return self.item_memory[item]['t_last'] is not None

    def review(self, item, recall, time):
        """Registers the event that the user reviewed an item at time 't'
        with the given recall."""

        self.item_memory[item]['t_last'] = time

        if recall == 1:
            self.item_memory[item]['right'] += 1
        else:
            self.item_memory[item]['wrong'] += 1


class ItemSelector:
    """This is the interface for ItemSelectors."""

    def __init__(self, seed, difficulty):
        self.seed = seed
        self.RS = np.random.RandomState(seed=seed)
        self.items = list(difficulty.keys())

    def is_fresh(self):
        """Returns whether the item selector has already been used."""
        return NotImplemented

    def get_session_items(
        self,
        time: float,
        memory_model: MemoryModel,
        session_size: int
    ) -> List[str]:
        return NotImplemented


class RRItemSelector(ItemSelector):
    def __init__(self, seed, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty
        )
        self.sorted_items = sorted(
            difficulty.keys(),
            key=lambda x: difficulty[x]
        )
        # TODO: Should use `m_0` instead of difficulty?
        # Cycle items from easy to difficult and back to easy
        self.item_generator = cycle(self.sorted_items)
        self._fresh = True

    def is_fresh(self):
        return self._fresh

    def get_session_items(self, time, memory_model, session_size):
        """Determine which items to select."""
        self._fresh = False
        return list(islice(self.item_generator, session_size))


class PlayBackSelector(ItemSelector):
    def __init__(self, seed, historical_sessions, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty,
        )
        self.start_time = historical_sessions['session_start'][0]
        self.historical_sessions = historical_sessions
        self.session_counter = 0

    def is_fresh(self):
        return self.session_counter == 0

    def get_session_items(self, time, memory_model, session_size):
        """Plays back the session."""
        sess_time = self.historical_sessions['session_start'][self.session_counter]
        sess_items = self.historical_sessions['session_items'][self.session_counter]

        assert sess_time - self.start_time == time, \
            f"Discrepancy between historical and session time at idx = {self.session_counter}"

        # Note that we will practice items only once even if they were selected
        # twice during a single session.
        assert len(sess_items) <= session_size, \
            f"Discrepancy between historical and session items at idx = {self.session_counter}"

        self.session_counter += 1
        return sess_items.keys()


class RandomizedSelector(ItemSelector):
    def __init__(self, seed, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty
        )
        self._fresh = True

    def is_fresh(self):
        return self._fresh

    def get_session_items(self, time, memory_model, session_size):
        self._fresh = False
        return self.RS.choice(
            self.items,
            size=min(session_size, len(self.items)),
            replace=False
        )


class SpacedItemSelector(ItemSelector):
    def __init__(self, seed, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty
        )
        self.num_items = len(self.items)
        self._fresh = True

    def is_fresh(self):
        return self._fresh

    def get_session_items(
        self,
        time: float,
        memory_model: MemoryModel,
        session_size: int
    ):
        """Select an item using spaced-selection algorithm.

        Is the true spaced-selection algorithm if the number of items is the
        same for each session.
        """
        one_by_sqrt_q_fixed = session_size / self.num_items

        self._fresh = False
        item_selection_prob = self.RS.rand(len(self.items))
        item_thresholds = [
            one_by_sqrt_q_fixed * (1 - memory_model.get_recall_prob(item, time=time))
            for item in self.items
        ]

        item_selected = [
            item
            for prob, threshold, item in zip(item_selection_prob,
                                             item_thresholds,
                                             self.items)
            if prob <= threshold
        ]
        return item_selected


class SpacedDynItemSelector(ItemSelector):
    def __init__(self, seed, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty
        )
        self.num_items = len(self.items)
        self._fresh = True

    def is_fresh(self):
        return self._fresh

    def get_session_items(
        self,
        time: float,
        memory_model: MemoryModel,
        session_size: int
    ):
        """Select an item using spaced-selection algorithm.

        However, the 'q' is changed dynamically such that the expected
        number of items is always session_size.
        """
        self._fresh = False

        item_selection_prob = self.RS.rand(len(self.items))
        item_raw_thresholds = [
            (1 - memory_model.get_recall_prob(item, time=time))
            for item in self.items
        ]
        sum_thresholds = np.sum(item_raw_thresholds)

        one_by_sqrt_q_dyn = session_size / sum_thresholds

        item_selected = [
            item
            for prob, threshold, item in zip(item_selection_prob,
                                             item_raw_thresholds,
                                             self.items)
            if prob <= one_by_sqrt_q_dyn * threshold
        ]
        return item_selected


class SpacedRankedItemSelector(ItemSelector):
    def __init__(self, seed, difficulty):
        super().__init__(
            seed=seed,
            difficulty=difficulty
        )
        self.num_items = len(self.items)
        self._fresh = True

    def is_fresh(self):
        return self._fresh

    def get_session_items(
        self,
        time: float,
        memory_model: MemoryModel,
        session_size: int
    ):
        """Select an item using spaced-selection ranking.

        This selects items with highest (1 - m(t)), i.e. lowest m(t), first.
        """
        self._fresh = False
        item_recall = sorted([
            (memory_model.get_recall_prob(item, time=time), item)
            for item in self.items
        ])

        # This automatically is upper bounded by the total number of items.
        item_selected = [item for (_recall, item) in item_recall[:session_size]]
        return item_selected


def mk_item_selector(item_selector_opts):
    """Create an item selector."""

    difficulty = item_selector_opts['difficulty']
    # m_0 = item_selector_opts['m_0']
    seed = item_selector_opts['seed']
    kind = item_selector_opts['kind']

    if kind == REPLAY_SELECTOR:
        return PlayBackSelector(
            seed=seed,
            historical_sessions=item_selector_opts['historical_sessions'],
            difficulty=difficulty
        )
    else:
        if kind == RANDOMIZED:
            return RandomizedSelector(
                seed=seed,
                difficulty=difficulty,
            )
        elif kind == ROUND_ROBIN:
            return RRItemSelector(
                seed=seed,
                difficulty=difficulty,
            )
        elif kind == SPACED_SELECTION:
            return SpacedItemSelector(
                seed=seed,
                difficulty=difficulty,
            )
        elif kind == SPACED_SELECTION_DYN:
            return SpacedDynItemSelector(
                seed=seed,
                difficulty=difficulty,
            )
        elif kind == SPACED_RANKING:
            return SpacedRankedItemSelector(
                seed=seed,
                difficulty=difficulty,
            )
        else:
            raise ValueError('Unknown model kind: ', kind)


class Teacher:
    def __init__(self, seed, memory_model_opts, item_selector_opts):
        self.RS = np.random.RandomState(seed=seed)
        self._item_selector_opts = item_selector_opts
        self._memory_model_opts = memory_model_opts

        self.memory_model = MemoryModel(**memory_model_opts)
        self.item_selector = mk_item_selector(item_selector_opts)

    def is_fresh(self):
        return self.item_selector.is_fresh()

    def get_session_items(self, time, session_size):
        """Return items to ask the user right now."""
        return self.item_selector.get_session_items(
            time=time,
            memory_model=self.memory_model,
            session_size=session_size
        )

    def record_recalls(self, recalls, time):
        """Record the recalls of the user for each item at the given time."""
        for item, recall in recalls.items():
            self.memory_model.review(item, recall=recall, time=time)


class Student:
    def __init__(self, seed, kind, memory_model_opts,
                 historical_sessions=None):
        self.RS = np.random.RandomState(seed=seed)
        self.kind = kind
        self.memory_model = MemoryModel(**memory_model_opts)
        self.historical_sessions = historical_sessions
        self.session_counter = 0

    def is_fresh(self):
        return self.session_counter == 0

    def get_recall_for(self, items, time, get_speculative_prob=False):
        """Try to recall the given items.

        If get_speculative_prob is True, then does not consider this as a
        review but as an exam.
        """
        recalls = {}
        for item in items:
            if self.kind != REPLAY or get_speculative_prob:
                recall_prob = self.memory_model.get_recall_prob(item, time=time)
            else:
                recall_prob = self.historical_sessions['session_items'][self.session_counter][item]

            if get_speculative_prob:
                recalls[item] = recall_prob
            else:
                recall = 1.0 if self.RS.rand() < recall_prob else 0.0
                recalls[item] = recall

                # Assuming a recall/review model, i.e. answers are given
                # immediately after a review
                self.memory_model.review(item=item, recall=recall, time=time)

        self.session_counter += 1
        return recalls


class Simulator:
    def __init__(
        self,
        seed,
        teaching_times,
        session_sizes,
        all_items,
        student,
        teacher,
    ):
        assert len(teaching_times) == len(session_sizes)

        self.RS = np.random.RandomState(seed=seed)
        self.seed = seed
        self.teaching_times = teaching_times
        self.session_sizes = session_sizes

        self.student = student
        self.teacher = teacher

        self.exam_items = all_items
        self.teaching_times = teaching_times

        self.item_attempts = defaultdict(int)
        self.sessions = []

    def run(self):
        # Have to simulate the training times
        assert self.student.is_fresh(), "(Re?)-running with a trained student."
        assert self.teacher.is_fresh(), "(Re?)-running with a trained teacher."

        for t, session_size in zip(self.teaching_times, self.session_sizes):
            items = self.teacher.get_session_items(time=t, session_size=session_size)
            recalls = self.student.get_recall_for(items=items, time=t)
            self.teacher.record_recalls(recalls, time=t)

            # Book-keeping
            self.sessions.append(recalls)
            for item in items:
                self.item_attempts[item] += 1

    def get_total_items_attempted(self):
        return sum(self.item_attempts.values())

    def get_all_sessions(self):
        return self.sessions

    def eval(self, exam_time):
        # Ask all questions
        return self.student.get_recall_for(items=self.exam_items,
                                           time=exam_time,
                                           get_speculative_prob=True)


def mk_student(student_opts):
    return Student(**student_opts)


def mk_teacher(teacher_opts):
    return Teacher(**teacher_opts)


def mk_simulator(
    seed,
    all_items,
    teaching_times,
    session_sizes,
    student_opts,
    teacher_opts
):
    student = mk_student(student_opts)
    teacher = mk_teacher(teacher_opts)
    return Simulator(
        seed=seed,
        all_items=all_items,
        student=student,
        teacher=teacher,
        teaching_times=teaching_times,
        session_sizes=session_sizes,
    )


@click.command()
@click.argument('difficulty_params')
@click.argument('user_sessions_csv')
@click.argument('sim_results_csv')
@click.option('--seed', default=42, help='Random seed for the experiment.', show_default=True)
@click.option('--difficulty-kind', default=HLR, help='Which memory model to assume for the difficulty_params.', show_default=True, type=click.Choice([HLR, POWER]))
@click.option('--student-kind', default=HLR, help='Which memory model to assume for the student.', show_default=True, type=click.Choice([HLR, POWER, REPLAY]))
@click.option('--teacher-kind', default=RANDOMIZED, help='Which teacher model to simulate.', show_default=True, type=click.Choice([RANDOMIZED, SPACED_SELECTION, REPLAY_SELECTOR, ROUND_ROBIN, SPACED_SELECTION_DYN, SPACED_RANKING]))
@click.option('--num-users', default=100, help='How many users to run the experiments for.', show_default=True)
# This option is no longer relevant.
# @click.option('--session-size', default=None, help='How many maximum questions should be asked (in expectation) each session?', show_default=True)
@click.option('--user-id', default=None, help='Which user to run the simulation for? [Runs for the user with maximum attempts otherwise.]', show_default=True)
@click.option('--force/--no-force', default=False, help='Whether to overwrite output file.', show_default=True)
def run(sim_results_csv, difficulty_params, user_sessions_csv, seed, num_users,
        difficulty_kind, student_kind, teacher_kind, user_id, force):
    """Run the simulation with the given output of training the memory model in
    the file DIFFICULTY_PARAMS weights file.

    It also reads the user session information from USER_SESSIONS_CSV to
    generate feasible teaching times.

    Finally, after running the simulations for 10-seeds, the results are saved
    in SIM_RESULTS_CSV.
    """

    if os.path.exists(sim_results_csv) and not force:
        print('{} exists and --force not supplied.'.format(sim_results_csv))
        sys.exit(1)

    df_attempts = pd.read_csv(user_sessions_csv)

    # TODO: Temporary fix; the permanent fix should be in hlr_learning.py
    df_attempts['lexeme_id'] = 'de:' + df_attempts['lexeme_id']

    difficulty_params = os.path.abspath(difficulty_params)

    memory_model_opts = read_difficulty(
        difficulty_params,
        df_attempts=df_attempts,
        kind=difficulty_kind,
        seed=seed * 7
    )

    all_items = memory_model_opts['difficulty'].keys()

    if user_id is None:
        # By default, just simulate the 100 most prolific users
        user_ids = (
            df_attempts
            .groupby('user_id')
            .size()
            .sort_values(ascending=False)
            .head(num_users)
            .reset_index()
            .user_id
            .to_list()
        )
    else:
        user_ids = [user_id]

    # Trick to make the function usable with Multiprocessing.
    # There is a way to make it work by sharing memory via MP.
    global _worker_user_id
    def _worker_user_id(params):
        (seed, user_id) = params

        op = []

        df_user = df_attempts[df_attempts.user_id == user_id]

        user_sessions = mk_sessions(
            df_user=df_user,
            threshold=FIVE_MINUTES,
            with_items=True,
        )

        # Teaching times always start from 0
        first_attempt_time = user_sessions['session_start'][0]
        teaching_times = (
            np.array(user_sessions['session_start']) -
            user_sessions['session_start'][0]
        )

        session_sizes = np.array(user_sessions['session_len'])

        for teacher_kind, student_kind in [
            (REPLAY_SELECTOR, REPLAY),
            (REPLAY_SELECTOR, HLR),
            (SPACED_RANKING, HLR),
            (SPACED_SELECTION_DYN, HLR),
            (RANDOMIZED, HLR),
            (ROUND_ROBIN, HLR)
        ]:
            item_selector_opts = {
                'difficulty': memory_model_opts['difficulty'],
                'm_0': memory_model_opts['m_0'],
                'seed': seed + 1001,
                'kind': teacher_kind,
                'historical_sessions': user_sessions,
            }

            student_memory_model_opts = change_memory_model_seed(
                memory_model_opts,
                new_seed=seed * 7
            )

            student_opts = {
                'kind': student_kind,
                'seed': seed * 13,
                'historical_sessions': user_sessions,
                'memory_model_opts': student_memory_model_opts,
            }

            teacher_memory_model_opts = change_memory_model_seed(
                memory_model_opts,
                new_seed=seed * 6,
            )
            teacher_opts = {
                'seed': seed * 101,
                'memory_model_opts': teacher_memory_model_opts,
                'item_selector_opts': item_selector_opts,
            }

            sim = mk_simulator(
                seed=seed + 1,
                all_items=all_items,
                teaching_times=teaching_times,
                student_opts=student_opts,
                teacher_opts=teacher_opts,
                session_sizes=session_sizes,
            )

            sim.run()
            days_1 = 24 * 60 * 60
            eval_1 = sim.eval(exam_time=teaching_times[-1] + days_1)       # Exam after 1 day
            eval_10 = sim.eval(exam_time=teaching_times[-1] + 10 * days_1)  # Exam after 10 days
            # perf_diff = {item: eval_1[item] - eval_2[item] for item in eval_1}
            # print('For student = {}, teacher = {}, attempts = {}'
            #       .format(student_kind, teacher_kind, sim.get_total_items_attempted()))
            # print('1 day = {}, 10 day = {}'
            #       .format(np.mean(list(eval_1.values())),
            #               np.mean(list(eval_2.values()))))

            op.append({
                'user_id': user_id,
                'eval_1': np.mean(list(eval_1.values())),
                'eval_10': np.mean(list(eval_10.values())),
                'seed': seed,
                'sessions': len(session_sizes),
                'attempts': sim.get_total_items_attempted(),
                'item_selector': teacher_opts['item_selector_opts']['kind'],
                'difficulty_params_file': difficulty_params,
                'student_model': student_opts['kind'],
                'first_attempt_time': first_attempt_time,
                'last_attempt_time': first_attempt_time + teaching_times[-1],
            })

            seed += 1

        return op

    all_seeds = np.arange(len(user_ids) * 10) + seed

    res = []
    try:
        with MP.Pool() as pool:
            finished = 0
            for y in pool.imap_unordered(
                    _worker_user_id,
                    zip(all_seeds, cycle(user_ids))
            ):
                res.extend(y)

                finished += 1
                if finished % 100 == 0:
                    print(f'{finished} ...', end='')

        print()
    except Exception as e:
        print('Error faced: ', e)

    res_df = pd.DataFrame(res)
    res_df.to_csv(sim_results_csv, index=False)
    print('Done.')


if __name__ == '__main__':
    run()
