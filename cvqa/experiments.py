import json
import numpy as np

import logging
import time
from itertools import product
from logging.handlers import RotatingFileHandler
from pathlib import Path

import inspect

import torch

class Experiments(object):

    @staticmethod
    def load_results(log_root):
        files = list(Path(log_root).glob('results.*'))

        results = []
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    results.append(json.loads(line))
        return results

    def __init__(self, model_builder, trainer, log_root=None, save_checkpoints=True):
        if log_root is None:
            log_root = 'experiments/run--' + time.strftime("%m-%d--%H-%M-%S")

        self.log_root = log_root
        self.model_builder = model_builder
        self.trainer = trainer
        self.save_checkpoints = save_checkpoints
        self.results = []
        self.loggers = {}
        self.log_enabled = True
        self.model_builder_vars = inspect.signature(model_builder).parameters.keys()
        self.trainer_vars = inspect.signature(trainer).parameters.keys()

        Path(log_root).mkdir(parents=True, exist_ok=True)

    def set_log_enabled(self, enabled):
        self.log_enabled = enabled

    def path(self, *parts):
        f = None
        if '.' in parts[-1]:
            f = parts[-1]
            parts = parts[:-1]

        p = Path(self.log_root, *parts)
        p.mkdir(parents=True, exist_ok=True)
        if f is not None:
            p = p / f

        return p

    def execute(self, hyper_params, limit=1000):
        considered_vars = set(self.model_builder_vars).union(set(self.trainer_vars))
        not_considered = [k for k in hyper_params if k not in considered_vars]
        if len(not_considered) > 0:
            raise ValueError(f'Some parameters are not accepted by model builder / trainer; {not_considered}')

        hp_vals_list = list(hyper_params.values())
        for i, hp_vals in enumerate(product(*hp_vals_list)):
            hp_map = {hp_key: hp_vals[j] for j, hp_key in enumerate(hyper_params)}
            try:
                build_params = {k: hp_map[k] for k in self.model_builder_vars if k in hp_map}
                model = self.model_builder(**build_params)

                train_params = {k: hp_map[k] for k in self.trainer_vars if k in hp_map}
                train_metrics = self.trainer(model, **train_params)
                if self.save_checkpoints:
                    checkpoint_path = self.path('model_checkpoints', f'model_{i}.pkl')
                    torch.save(model.state_dict(), checkpoint_path)
            except Exception as ex:
                train_metrics = {
                    'error': str(ex)
                }
            self.report_experiment(i, hp_map, train_metrics)
            if i % 10 == 0:
                self.roll_logs()
            if i > limit:
                break

        self.roll_logs()

    def report_experiment(self, id, hp_map, train_metrics):
        report = {
            'id': id,
            'hyper_parameters': hp_map,
            'train_metrics': train_metrics
        }
        self.results.append(report)
        if self.log_enabled:
            self.get_logger().info(json.dumps(report, cls=NpEncoder))

    def roll_logs(self):
        for log in self.loggers.values():
            log.handlers[0].doRollover()

    def get_logger(self, name='results'):
        if name not in self.loggers:
            logger = logging.getLogger(f'experiments.{self.log_root}.{name}')
            logger.setLevel(logging.DEBUG)

            log_file = self.path(f'{name}.jsonl')

            # fh = logging.FileHandler(log_file)
            fh = RotatingFileHandler(log_file, backupCount=100)
            fh.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.propagate = False
            self.loggers[name] = logger

        return self.loggers[name]


class NpEncoder(json.JSONEncoder):
    """
    Copied from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)