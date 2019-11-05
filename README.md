# Spaced-Selection 

_Spaced Selection_ is a method for optimally selecting the items which the user should revise during a given session to optimize learning.

The modeling of human memory is based on our previous work, [Memorize](https://github.com/Networks-Learning/memorize), but instead of choosing the optimal time to review each item, in this work, we allow the user to select the session time and we choose the set of items which she will study during the session.

This repository consists of scripts for analysis of Spaced selection and baseline as well as code to run simulations to compare the performance of different item selection strategies.

The model was trained using data from the popular smart driving-learning app by [Swift](https://www.swift.ch/). The data on which the models were trained is not available. A preprint will be shortly made available.

To prepare, download all the `csv` files to the `data/` folder. 
Unless otherwise stated, the code should be run from the root folder.

## Installing Dependencies

```bash
pip install -r code/requirements.txt
```

## Swift data to HLR format

```
➔ ./swift_to_hlr.py --help
Usage: swift_to_hlr.py [OPTIONS] INPUT_DIR OUTPUT_HLR_CSV OUTPUT_SIM_CSV

  Convert CSV files in INPUT_DIR from Swift.ch format to Duolingo's HLR
  format and save in OUTPUT_HLR_CSV, as well as to extract each attempt and
  save in OUTPUT_SIM_CSV.

Options:
  --verbose / --no-verbose  Verbose output.  [default: True]
  --force / --no-force      Overwrite output.  [default: False]
  --min-count INTEGER       Minimum number of times a user must have practiced
                            a question to include it for training/prediction.
                            [default: 1]
  --results-dir TEXT        The results folder for Lineage.  [default:
                            results]
  --help                    Show this message and exit.
```

The `processed` folder contains an example of learned difficulty parameters for
the HLR model. However, the user sessions file is not included with the
repository.


## HLR Parameter learning

```
➔ ./hlr_learning.py --help
usage: hlr_learning.py [-h] [-b] [-l] [-t] [-m METHOD] [-x MAX_LINES]
                       [-h_reg HLWT] [-l2wt L2WT] [-bins BINS]
                       [-epochs EPOCHS] [-shuffle SHUFFLE]
                       [-training_fraction TRAINING_FRACTION] [-l_rate L_RATE]
                       [-o OUTPUT_FOLDER]
                       input_file

Fit a SpacedRepetitionModel to data.

positional arguments:
  input_file            log file for training

optional arguments:
  -h, --help            show this help message and exit
  -b                    omit bias feature
  -l                    omit lexeme features
  -t                    omit half-life term
  -m METHOD             hlr, lr, leitner, pimsleur, hlr-pw, power
  -x MAX_LINES          maximum number of lines to read (for dev)
  -h_reg HLWT           h regularization weight
  -l2wt L2WT            L2 regularization weight
  -bins BINS            File where the bins boundaries are stored (in days).
  -epochs EPOCHS        Number of epochs to train for.
  -shuffle SHUFFLE      The seed to use to shuffle data, -1 for no shuffling.
  -training_fraction TRAINING_FRACTION
                        The fraction of data to use for training.
  -l_rate L_RATE        Where to save the results.
  -o OUTPUT_FOLDER      Where to save the results.
```

### Grid execution

This is a side script for executing the model on a SLURM engine, if one is available, for easy parameter search.

```
➔ ./slurm/grid_search_run.py --help
Usage: grid_search_run.py [OPTIONS] INPUT_CSV OUTPUT_DIR

Options:
  --slurm-output-dir TEXT  Where to save the output  [default: slurm-output]
  --dry / --no-dry         Dry run.  [default: True]
  --epochs INTEGER         Epochs.  [default: 500]
  --mem INTEGER            How much memory will each job need (MB).  
                           [default: 10000]
  --timeout INTEGER        Minutes to timeout.
  --shuffle INTEGER        Seed to shuffle training/testing using.
  --l-rate FLOAT           Initial learning rate.
  --help                   Show this message and exit.
```

## HLR model evaluation

```
➔ ./hlr_eval.py --help
Usage: hlr_eval.py [OPTIONS] RESULTS_DIR OUTPUT_CSV

  Read all *.detailed files from RESULTS_DIR, calculate the metrics, and
  save output to OUTPUT_CSV.

Options:
  --debug / --no-debug  Run in single threaded mode for debugging.
  --help                Show this message and exit.
```

## Simulation

```
➔ ./simulation.py --help
Usage: simulation.py [OPTIONS] DIFFICULTY_PARAMS USER_SESSIONS_CSV
                     SIM_RESULTS_CSV

  Run the simulation with the given output of training the memory model in
  the file DIFFICULTY_PARAMS weights file.

  It also reads the user session information from USER_SESSIONS_CSV to
  generate feasible teaching times.

  Finally, after running the simulations for 10-seeds, the results are saved
  in SIM_RESULTS_CSV.

Options:
  --seed INTEGER                  Random seed for the experiment.  [default: 42]
  --difficulty-kind [HLR|POWER]   Which memory model to assume for the
                                  difficulty_params.  [default: HLR]
  --student-kind [HLR|POWER|REPLAY]
                                  Which memory model to assume for the
                                  student.  [default: HLR]
  --teacher-kind [RANDOMIZED|SPACED_SELECTION|REPLAY_SELECTOR|ROUND_ROBIN|SPACED_SELECTION_DYN|SPACED_RANKING]
                                  Which teacher model to simulate.  
                                  [default: RANDOMIZED]
  --num-users INTEGER             How many users to run the experiments for.
                                  [default: 100]
  --user-id TEXT                  Which user to run the simulation for? [Runs
                                  for the user with maximum attempts
                                  otherwise.]
  --force / --no-force            Whether to overwrite output file.  
                                  [default: False]
  --help                          Show this message and exit.
```

The required files `DIFFICULTY_PARAMS` (an example included in `processed/`
folder) and `USER_SESSIONS_CSV` are produced by the `swift_to_hlr.py` script above.

The different version of the _Spaced-selection_ algorithm which can be
simulated are:

 - `SPACED_RANKING` chooses the top-`k` items in terms of forgetting probability (that depends on the current half-life factor) for each session deterministically, where `k` can be tuned/modified per session (produced by the simulator or sampled from real data).

 - `SPACED_SELECTION_DYN` chooses the `k` items probabilistically with each item's selection proportional to the probability of forgetting it (that depends on the current half-life factor) for each session, where `k` can be tuned/modified per session  (produced by the simulator or sampled from real data).

- `SPACED_SELECTION` samples `k` items at random proportionally to the forgetting probability (that depends on the current half-life factor) for each session, where `k` is set by the population average size of sessions.
