#!/usr/bin/env python
"""
Copyright (c) 2016 Duolingo Inc. MIT Licence.

Python script that implements spaced repetition models from Settles & Meeder (2016).
Recommended to run with pypy for efficiency. See README.
"""

import argparse
import csv
import gzip
import math
import os
import random
import sys
import warnings
from sys import intern
from collections import defaultdict, namedtuple
from sklearn.metrics import roc_auc_score

from typing import List, Optional

# from dataworkspaces.lineage import LineageBuilder

# various constraints on parameters and outputs
MIN_HALF_LIFE = 1.0 / (24 * 60)  # 1 minute
MAX_HALF_LIFE = 274.0  # 9 months
LN2 = math.log(2.0)


# data instance object
feature_names: List[str] = "p t fv h a lang right wrong ts uid lexeme".split()
Instance = namedtuple("Instance", feature_names)


class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements the following approaches:
      - 'hlr' (half-life regression; trainable)
      - 'lr' (logistic regression; trainable)
      - 'power' (power-law regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
    """

    def __init__(
        self,
        method="hlr",
        omit_h_term=False,
        initial_weights=None,
        lrate=0.001,
        hlwt=0.1,
        l2wt=0.1,
        sigma=1.0,
    ):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)

        if method == "power":
            self.weights["A"] = 1.0
            self.weights["B"] = 1.0

        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma

    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k] * x_k for (k, x_k) in inst.fv])
            return hclip(base ** dp)
        except Exception as e:
            warnings.warn("hClip warning: " + repr(e), RuntimeWarning)
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.0):
        if self.method == "power":
            h = self.halflife(inst, base)
            p = self.weights["A"] * (1 + self.weights["B"] * inst.t) ** (-1 / h)
            return pclip(p), h
        elif self.method == "hlr" or self.method == "hlr-pw":
            h = self.halflife(inst, base)
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "leitner":
            try:
                h = hclip(2.0 ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "pimsleur":
            try:
                h = hclip(2.0 ** (2.35 * inst.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "lr":
            dp = sum([self.weights[k] * x_k for (k, x_k) in inst.fv])
            p = 1.0 / (1 + math.exp(-dp))
            return pclip(p), random.random()
        else:
            raise Exception

    def train_update(self, inst):
        if self.method == "power":
            p, h = self.predict(inst)
            A, B = self.weights["A"], self.weights["B"]
            log_p = math.log(inst.p)
            h_empirical = hclip(-math.log(A * (1 + B * inst.t)) / log_p)
            dlp_dw = 2.0 * (p - inst.p) * (math.log(1 + B * inst.t)) / h * LN2 * p
            dlh_dw = 2.0 * (h - h_empirical) * LN2 * h

            dlp_dA = 2 * (p - inst.p) * (1 + B * inst.t) ** (-1 / h)
            dlp_dB = 2 * (p - inst.p) * p * inst.t / (h * (1 + B * inst.t))

            dlh_dA = 2 * (h - h_empirical) / (A * log_p)
            dlh_dB = 2 * (h - h_empirical) * inst.t / ((1 + B * inst.t) * log_p)

            for (k, x_k) in inst.fv + [("A", None), ("B", None)]:
                rate = (
                    (1.0 / (1 + inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                )

                if k == "A":
                    self.weights[k] -= rate * dlp_dA
                    if not self.omit_h_term:
                        self.weights[k] -= rate * self.hlwt * dlh_dA

                    self.weights[k] = max(0.0001, self.weights[k])
                elif k == "B":
                    self.weights[k] -= rate * dlp_dB
                    if not self.omit_h_term:
                        self.weights[k] -= rate * self.hlwt * dlh_dB

                    self.weights[k] = max(0.0001, self.weights[k])
                else:
                    self.weights[k] -= rate * dlp_dw * x_k

                    if not self.omit_h_term:
                        self.weights[k] -= rate * self.hlwt * dlh_dw * x_k

                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2

                # Increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == "hlr" or self.method == "hlr-pw":
            base = 2.0
            p, h = self.predict(inst, base)
            dlp_dw = 2.0 * (p - inst.p) * (LN2 ** 2) * p * (inst.t / h)
            dlh_dw = 2.0 * (h - inst.h) * LN2 * h
            for (k, x_k) in inst.fv:
                rate = (
                    (1.0 / (1 + inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                )
                # rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2
                # increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == "leitner" or self.method == "pimsleur":
            pass
        elif self.method == "lr":
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                # rate = (1./(1+inst.p)) * self.lrate   / math.sqrt(1 + self.fcounts[k])
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2
                # increment feature count for learning rate
                self.fcounts[k] += 1

    def train(self, trainset, testset, n_epochs=3):
        if self.method == "leitner" or self.method == "pimsleur":
            return
        random.shuffle(trainset)
        for epoch in range(n_epochs):
            print("epoch %d" % epoch)
            for inst in trainset:
                self.train_update(inst)
            self.eval(testset, prefix="epoch_eval")

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p) ** 2
        slh = (inst.h - h) ** 2
        return slp, slh, p, h

    def eval(self, testset, prefix=""):
        results = {"p": [], "h": [], "pp": [], "hh": [], "slp": [], "slh": []}
        for inst in testset:
            slp, slh, p, h = self.losses(inst)
            results["p"].append(inst.p)  # ground truth
            results["h"].append(inst.h)
            results["pp"].append(p)  # predictions
            results["hh"].append(h)
            results["slp"].append(slp)  # loss function values
            results["slh"].append(slh)
        mae_p = mae(results["p"], results["pp"])
        mae_h = mae(results["h"], results["hh"])
        cor_p = spearmanr(results["p"], results["pp"])
        cor_h = spearmanr(results["h"], results["hh"])
        auc_p = roc_auc_score([round(x) for x in results["p"]], results["pp"])
        total_slp = sum(results["slp"])
        total_slh = sum(results["slh"])
        total_l2 = sum([x ** 2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt * total_slh + self.l2wt * total_l2
        if prefix:
            sys.stderr.write("%s\t" % prefix)
        sys.stderr.write(
            "%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor(h)=%.3f\tauc(p)=%.3f\n"
            % (
                total_loss,
                total_slp,
                self.hlwt * total_slh,
                self.l2wt * total_l2,
                mae_p,
                cor_p,
                mae_h,
                cor_h,
                auc_p,
            )
        )

    def dump_weights(self, fname):
        with open(fname, "w") as f:
            for (k, v) in self.weights.items():
                f.write("%s\t%.4f\n" % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, "w") as f:
            f.write("p\tpp\th\thh\tlang\tuser_id\ttimestamp\n")
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write(
                    "%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\n"
                    % (inst.p, pp, inst.h, hh, inst.lang, inst.uid, inst.ts)
                )

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, "w") as f:
            f.write("p\tpp\th\thh\tlang\tuser_id\ttimestamp\tlexeme_tag\n")
            for inst in testset:
                pp, hh = self.predict(inst)
                for i in range(inst.right):
                    f.write(
                        "1.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n"
                        % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme)
                    )
                for i in range(inst.wrong):
                    f.write(
                        "0.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n"
                        % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme)
                    )


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.00001), 0.99999)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def mean(lst):
    # the average of a list
    return float(sum(lst)) / len(lst)


def spearmanr(l1, l2):
    # spearman rank correlation
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.0
    d1 = 0.0
    d2 = 0.0
    if len(l1) > 0:
        for i in range(len(l1)):
            num += (l1[i] - m1) * (l2[i] - m2)
            d1 += (l1[i] - m1) ** 2
            d2 += (l2[i] - m2) ** 2
        return num / math.sqrt(d1 * d2)
    else:
        return float("nan")


def read_data(
    input_file,
    method,
    omit_bias=False,
    omit_lexemes=False,
    max_lines=None,
    bins=None,
    seed=-1,
    training_fraction=1.0,
):
    # read learning trace data in specified format, see README for details
    sys.stderr.write("reading data...")

    if method == "hlr-pw":
        num_quantiles = len(bins) - 1
        quantile_intervals = list(zip(bins[:-1], bins[1:]))
    else:
        num_quantiles, quantile_intervals = None, []

    instances = list()
    if input_file.endswith("gz"):
        f = gzip.open(input_file, "rb")
    else:
        f = open(input_file, "r")
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row["p_recall"]))
        t = float(row["delta"]) / (60 * 60 * 24)  # convert time delta to days
        h = hclip(-t / (math.log(p, 2)))
        lang = "%s->%s" % (row["ui_language"], row["learning_language"])
        # lexeme_id = row['lexeme_id']
        lexeme_string = row["lexeme_string"]
        timestamp = int(row["timestamp"])
        user_id = row["user_id"]
        seen = int(row["history_seen"])
        right = int(row["history_correct"])
        wrong = seen - right
        right_this = int(row["session_correct"])
        wrong_this = int(row["session_seen"]) - right_this
        # feature vector is a list of (feature, value) tuples
        fv = []
        # core features based on method
        if method == "leitner":
            fv.append((intern("diff"), right - wrong))
        elif method == "pimsleur":
            fv.append((intern("total"), right + wrong))
        elif method == "hlr" or method == "power":
            fv.append((intern("right"), right))
            fv.append((intern("wrong"), wrong))
            # fv.append((intern('right'), math.sqrt(1+right)))
            # fv.append((intern('wrong'), math.sqrt(1+wrong)))
        elif method == "hlr-pw":
            # Now need to fill in the right_{quantile} for each row.
            for q in range(num_quantiles):
                in_this_quantile = (
                    quantile_intervals[q][0] <= t < quantile_intervals[q][1]
                )
                fv.append(("right_%d" % q, right if in_this_quantile else 0))
                fv.append(("wrong_%d" % q, wrong if in_this_quantile else 0))
        else:
            raise Exception("Unknown method {}".format(method))

        # optional flag features
        if method == "lr":
            fv.append((intern("time"), t))
        if not omit_bias:
            fv.append((intern("bias"), 1.0))
        if not omit_lexemes:
            # fv.append((intern('%s:%s' % (row['learning_language'], lexeme_string)), 1.))
            # Remove the 'de:' prefix.
            fv.append((intern(lexeme_string), 1.0))
        instances.append(
            Instance(
                p,
                t,
                fv,
                h,
                (right + 2.0) / (seen + 4.0),
                lang,
                right_this,
                wrong_this,
                timestamp,
                user_id,
                lexeme_string,
            )
        )
        if i % 1000000 == 0:
            sys.stderr.write("%d..." % i)
    sys.stderr.write("done!\n")
    splitpoint = int(0.9 * len(instances))

    if seed > 0:
        sys.stderr.write("Shuffling with seed %d.\n" % seed)
        random.seed(seed)
        random.shuffle(instances)

    training = instances[: int(splitpoint * training_fraction)]
    testing = instances[splitpoint:]

    return training, testing


argparser = argparse.ArgumentParser(description="Fit a SpacedRepetitionModel to data.")
argparser.add_argument(
    "-b", action="store_true", default=False, help="omit bias feature"
)
argparser.add_argument(
    "-l", action="store_true", default=False, help="omit lexeme features"
)
argparser.add_argument(
    "-t", action="store_true", default=False, help="omit half-life term"
)
argparser.add_argument(
    "-m",
    action="store",
    dest="method",
    default="hlr",
    help="hlr, lr, leitner, pimsleur, hlr-pw, power",
)
argparser.add_argument(
    "-x",
    action="store",
    dest="max_lines",
    type=int,
    default=None,
    help="maximum number of lines to read (for dev)",
)
argparser.add_argument(
    "-h_reg",
    action="store",
    dest="hlwt",
    type=float,
    help="h regularization weight",
    default=0.01,
)
argparser.add_argument(
    "-l2wt",
    action="store",
    dest="l2wt",
    type=float,
    help="L2 regularization weight",
    default=0.1,
)
argparser.add_argument(
    "-bins",
    action="store",
    dest="bins",
    help="File where the bins boundaries are stored (in days).",
    default=None,
)
argparser.add_argument(
    "-epochs",
    action="store",
    dest="epochs",
    type=int,
    help="Number of epochs to train for.",
    default=3,
)
argparser.add_argument(
    "-shuffle",
    action="store",
    dest="shuffle",
    type=int,
    default=-1,
    help="The seed to use to shuffle data, -1 for no shuffling.",
)
argparser.add_argument(
    "-training_fraction",
    action="store",
    dest="training_fraction",
    type=float,
    default=1.0,
    help="The fraction of data to use for training.",
)
argparser.add_argument(
    "-l_rate",
    action="store",
    dest="l_rate",
    type=float,
    default=0.001,
    help="Where to save the results.",
)
argparser.add_argument(
    "-o",
    action="store",
    dest="output_folder",
    type=str,
    default="results/",
    help="Where to save the results.",
)

argparser.add_argument("input_file", action="store", help="log file for training")

if __name__ == "__main__":

    # Show warnings only once
    warnings.simplefilter("once")

    args = argparser.parse_args()

    # model diagnostics
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.b:
        sys.stderr.write("--> omit_bias\n")
    if args.l:
        sys.stderr.write("--> omit_lexemes\n")
    if args.t:
        sys.stderr.write("--> omit_h_term\n")

    if args.method == "hlr-pw" and args.bins is None:
        sys.stderr.write(
            "Must provide a file with the time-bins for HLR/Piecewise-Constant rates."
        )
        sys.exit(-1)

    # read data set
    bins: Optional[List[float]]
    if args.bins:
        with open(args.bins) as bins_file:
            bins = [float(x) for x in bins_file]
    else:
        bins = None

    # builder = (
    #     LineageBuilder()
    #     .as_script_step()
    #     .with_parameters(vars(args))
    #     .with_input_path(args.input_file)
    # )

    # with builder.eval() as lineage:
    if True:
        trainset, testset = read_data(
            args.input_file,
            args.method,
            args.b,
            args.l,
            args.max_lines,
            bins=bins,
            seed=args.shuffle,
            training_fraction=args.training_fraction,
        )
        sys.stderr.write("|train| = %d\n" % len(trainset))
        sys.stderr.write("|test|  = %d\n" % len(testset))

        # train model & print preliminary evaluation info
        model = SpacedRepetitionModel(
            method=args.method,
            omit_h_term=args.t,
            hlwt=args.hlwt,
            l2wt=args.l2wt,
            lrate=args.l_rate,
        )
        model.train(trainset, testset, n_epochs=args.epochs)
        model.eval(testset, "test")

        excluded_filebits = {"input_file", "output_folder"}

        # write out model weights and predictions
        filebits = (
            [args.method]
            + [
                "{}-{}".format(k, v)
                if type(v) is not float
                else "{}-{:.9f}".format(k, v)
                for k, v in sorted(vars(args).items())
                if k not in excluded_filebits
            ]
            + [
                os.path.splitext(os.path.basename(args.input_file).replace(".gz", ""))[
                    0
                ]
            ]
        )

        if bins is not None:
            filebits += ["{}-bins".format(len(bins) - 1)]

        if args.max_lines is not None:
            filebits.append(str(args.max_lines))

        filebase = ",".join(filebits)
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        weights_file = os.path.join(args.output_folder, filebase + ".weights")
        # lineage.add_output_path(weights_file)
        model.dump_weights(weights_file)

        pred_file = os.path.join(args.output_folder, filebase + ".preds")
        # lineage.add_output_path(pred_file)
        model.dump_predictions(pred_file, testset)

        detailed_pred_file = os.path.join(args.output_folder, filebase + ".detailed")
        # lineage.add_output_path(detailed_pred_file)
        model.dump_detailed_predictions(detailed_pred_file, testset)
