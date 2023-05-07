
import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

import dataloader
import hparams_registry
import algorithm
import misc

import subprocess
import tqdm
import shlex

class Job:
    def __init__(self, train_args, models_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(models_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', 'train.py']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        print("jobs are: ",jobs)
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        print(commands)
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def multi_gpu_launcher(commands):
 
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training for different seeds')
    parser.add_argument('--algorithms', nargs='+', type=str, default='ERM_SMA')
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    args = parser.parse_args()
    algos = []
    algos.append(args.algorithms)

    args_list = []
    for trial_seed in range(args.n_trials):
        for algorithm in algos:
            all_test_envs = [[i] for i in range(4)]#[[1]]#
            for test_envs in all_test_envs:
                for hparams_seed in range(0, args.n_hparams):
                    train_args = {}
                    train_args['dataset'] = 'TerraIncognita'
                    train_args['algorithm'] = algorithm
                    train_args['test_envs'] = test_envs
                    train_args['holdout_fraction'] = args.holdout_fraction
                    train_args['hparams_seed'] = hparams_seed
                    train_args['data_dir'] = args.data_dir
                    train_args['trial_seed'] = trial_seed
                    train_args['seed'] = misc.seed_hash('TerraIncognita',
                        algorithm, test_envs, hparams_seed, trial_seed)
                    if args.hparams is not None:
                        train_args['hparams'] = args.hparams
                    args_list.append(train_args)

    print(args)
    print(args_list)
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    print('command launch executed')
    print("Launching the jobs")
    to_launch = [j for j in jobs]
    print(f'About to launch {len(to_launch)} jobs.')
    Job.launch(to_launch, multi_gpu_launcher)


    """
    def make_args_list(n_trials, algorithms, n_hparams, steps,
    data_dir, task, holdout_fraction, hparams):
    args_list = []
    for trial_seed in range(n_trials):
        for algorithm in algorithms:
            all_test_envs = [[i] for i in range(4)]#[[1]]#
            for test_envs in all_test_envs:
                for hparams_seed in range(0, n_hparams):
                    train_args = {}
                    train_args['dataset'] = 'TerraIncognita'
                    train_args['algorithm'] = algorithm
                    train_args['test_envs'] = test_envs
                    train_args['holdout_fraction'] = holdout_fraction
                    train_args['hparams_seed'] = hparams_seed
                    train_args['data_dir'] = data_dir
                    train_args['trial_seed'] = trial_seed
                    train_args['seed'] = misc.seed_hash('TerraIncognita',
                        algorithm, test_envs, hparams_seed, trial_seed)
                    if steps is not None:
                        train_args['steps'] = steps
                    if hparams is not None:
                        train_args['hparams'] = hparams
                    args_list.append(train_args)
    return args_list
    """