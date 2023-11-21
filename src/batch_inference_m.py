#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging
from sys import stdin, stdout
import yaml
import gc

import torch

from probing.inference import Inference


class NotAnExperimentDir(ValueError):
    pass


def find_last_model(experiment_dir):
    model_pre = os.path.join(experiment_dir, 'model')
    if os.path.exists(model_pre):
        return model_pre
    saves = filter(lambda f: f.startswith(
        'model.epoch_'), os.listdir(experiment_dir))
    last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
    return os.path.join(experiment_dir, last_epoch)

def find_inference_file_name(experiment_dir, prefix, data_dir):
    cfg = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(cfg):
        raise NotAnExperimentDir(f"{cfg} does not exist")
    with open(cfg) as f:
        train_fn = yaml.load(f, Loader=yaml.FullLoader)['train_file']
    inf = train_fn.replace('morphology_probes/data', 'datasets/'+data_dir+'/morphology_probes/data')
    task = inf.split('/')[-3]
    lang = inf.split('/')[-2]
    inf = inf.replace('/train', f'/{prefix}')
    outf = os.path.join(experiment_dir, f'{prefix}.out')
    accf = os.path.join(experiment_dir, f'{prefix}.word_accuracy')
    return inf, outf, accf, task, lang

def compute_accuracy(reference, prediction):
    acc = 0
    samples = 0
    with open(reference) as r, open(prediction) as p:
        for rline in r:
            try:
                pline = next(p)
            except StopIteration:
                logging.error(f"Prediction file {prediction} shorter "
                              f"than reference {reference}")
                return acc / samples
            if not rline.strip() and not pline.strip():
                continue
            rlabel = rline.rstrip("\n").split("\t")[3]
            plabel = pline.rstrip("\n").split("\t")[3]
            acc += (rlabel == plabel)
            samples += 1
    return acc / samples


def parse_args():
    p = ArgumentParser()
    p.add_argument("experiment_dirs", nargs="+", type=str,
                   help="Experiment directory")
    p.add_argument("data_path", nargs="+", type=str,
                   help="Inference data directory")
    p.add_argument("--slice", default="test", type=str)
    p.add_argument("--max-samples", default=None, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_path[0].split('/')[-3]
    for experiment_dir in args.experiment_dirs:
        if not os.path.isdir(experiment_dir):
            logging.info(f"{experiment_dir} not directory, skipping")
            continue
        try:
            slice = args.slice
            test_in, test_out, test_acc, task, lang = find_inference_file_name(experiment_dir, slice, data_dir)
            logging.info(f"Running inference on {task} probe on {data_dir} data with {slice} data slice") #find_parallel_perturbed_file_name
            try:
                inf = Inference(experiment_dir, test_in, max_samples=args.max_samples)
            except FileNotFoundError:
                logging.info(f"{test_in}: no such file")
                continue
            with open(test_out, 'w') as f:
                inf.run_and_print(f)
            acc = compute_accuracy(test_in, test_out)
            print(f"{task}\t{lang}\t{acc}")
            logging.info(f"{experiment_dir} test acc: {acc}")
            with open(test_acc, 'w') as f:
                f.write(f"{acc}\n")
            gc.collect()
            torch.cuda.empty_cache()
        except NotAnExperimentDir:
            logging.info(f"{experiment_dir}: no config.yaml, skipping")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

















