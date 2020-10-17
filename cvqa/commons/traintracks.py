import time
import logging
import os

from logging.handlers import RotatingFileHandler

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np


class TrainTracker(object):

    def __init__(self, root='tmp/traintracks', append_date=True, mem_threshold=200000):
        if append_date:
            root += time.strftime("--%m-%d--%H-%M-%S")

        self.root = root
        self.counters = {}
        self.metrics = {}
        self.loggers = {}
        self.plots = {}
        self.default_trigger_period = None
        self.default_trigger_counter = None
        self.triggers = []
        self.mem_threshold = mem_threshold
        self.mem_max_overflow = int(.5 * mem_threshold)
        Path(root).mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        f = None
        if '.' in parts[-1]:
            f = parts[-1]
            parts = parts[:-1]

        p = Path(self.root, *parts)
        p.mkdir(parents=True, exist_ok=True)
        if f is not None:
            p = p / f

        return p

    def moving_average(self, a, n=10):
        n = min(n, len(a))
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        normalizer = np.concatenate([np.arange(n) + 1, np.ones(len(a) - n) * n])
        return ret / normalizer

    def log_conf(self, conf_obj):
        with open(self.path('conf.json'), 'w') as outfile:
            json.dump(conf_obj, outfile, indent=4)

    def save_module(self, module):
        module_path = module.__file__
        module_file_name = os.path.basename(module_path)
        copyfile(module_path, self.path('saved-modules', module_file_name))

    def start(self, counter_name):
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
            if not self.default_trigger_counter:
                self.default_trigger_counter = counter_name

        self.counters[counter_name] += 1
        for t in self.triggers:
            if t['counter'] == counter_name and self.counters[counter_name] % t['period'] == 0:
                t['cb']()

        # Write plots if needed
        for metric_name, plot_spec in self.plots.items():
            period = plot_spec['freq_period']
            counter = plot_spec['freq_counter']
            if counter_name == counter and self.counters[counter_name] % period == 0:
                self.write_plot(metric_name, self.plots[metric_name], self.counters[counter_name])

    def get_logger(self, name):
        if name not in self.loggers:
            logger = logging.getLogger(f'traintracks.{name}')
            logger.setLevel(logging.DEBUG)

            log_file = self.path('metrics', f'{name}.jsonl')

            # fh = logging.FileHandler(log_file)
            fh = RotatingFileHandler(log_file, backupCount=100)
            fh.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.propagate = False
            self.loggers[name] = logger

        return self.loggers[name]

    def log_metric(self, metric_name, val):
        if metric_name not in self.metrics:
            m_data = {}
            for c in self.counters:
                m_data[c] = []

            m_data[metric_name] = []
            self.metrics[metric_name] = m_data
        else:
            m_data = self.metrics[metric_name]

        m_data[metric_name].append(val)
        log_json = {}
        for c in self.counters:
            c_val = self.counters[c]
            if c in m_data:
                m_data[c].append(c_val)
                log_json[c] = c_val
        log_json[metric_name] = val

        self.get_logger(metric_name).info(json.dumps(log_json))

        # Delete old data
        if len(m_data[metric_name]) > self.mem_threshold + self.mem_max_overflow:
            for k in m_data:
                del m_data[k][:self.mem_max_overflow]

    def get_metric(self, metric_name, movingavg=None):
        m_vec = self.metrics[metric_name][metric_name]

        if movingavg is not None:
            m_val = np.mean(m_vec[-movingavg:])
        else:
            m_val = m_vec[-1]

        return m_val

    def set_default_trigger(self, trigger_period, trigger_counter):
        self.default_trigger_period = trigger_period
        self.default_trigger_counter = trigger_counter

    def add_callback(self, cb, trigger_period=None, trigger_counter=None):
        self.triggers.append({
            'counter': trigger_counter or self.default_trigger_counter,
            'period': trigger_period or self.default_trigger_period,
            'cb': cb
        })

    def logs_rollover(self, period, counter):
        def roll_all_logs():
            for log in self.loggers.values():
                log.handlers[0].doRollover()

        self.add_callback(roll_all_logs, period, counter)

    def get_plot_spec(self, trigger_period=None, trigger_counter=None, x_axis=None, movingavg=None, last_n=None):
        trigger_period = trigger_period or self.default_trigger_period
        trigger_counter = trigger_counter or self.default_trigger_counter

        if x_axis is None:
            x_axis = trigger_counter

        return {
                'freq_period': trigger_period,
                'freq_counter': trigger_counter,
                'movingavg': movingavg,
                'last_n': last_n,
                'x_axis': x_axis
            }

    def add_plot(self, metric, trigger_period=None, trigger_counter=None, x_axis=None, movingavg=None, last_n=None):
        if metric not in self.plots:
            self.plots[metric] = self.get_plot_spec(trigger_period, trigger_counter, x_axis, movingavg, last_n)

    def plot(self, metric, x_axis=None, movingavg=None, last_n=None):
        p_spec = self.get_plot_spec(x_axis=x_axis, movingavg=movingavg, last_n=last_n)
        counter_val = self.counters[p_spec['freq_counter']]
        self.write_plot(metric, p_spec, counter_val)

    def write_plot(self, metric_name, plot_spec, counter_val):

        x_axis_label = plot_spec['x_axis']

        if not metric_name in self.metrics:
            return

        y_data = self.metrics[metric_name][metric_name]
        x_data = self.metrics[metric_name][x_axis_label]
        last_n = plot_spec['last_n']

        fig = plt.figure(figsize=(8, 5))
        if type(y_data[0]) == list:
            if last_n:
                y_data = y_data[-last_n:]
                x_data = x_data[-last_n:]

            i = 0
            for ys in y_data:
                xs = (x_data[i],) * len(ys)
                plt.plot(xs, ys, 'o', color='C0', alpha=.5)
                i += 1
        else:
            movingavg = plot_spec['movingavg']

            if movingavg is not None:
                y_data = self.moving_average(y_data, n=movingavg)
            if last_n:
                y_data = y_data[-last_n:]
                x_data = x_data[-last_n:]

            plt.plot(x_data, y_data, label=metric_name)
            plt.xlabel(x_axis_label)
            plt.legend()

        plot_file = self.path(f'plots-{metric_name}', f'plt_{metric_name}_{x_axis_label}={counter_val:07}.png')
        fig.savefig(plot_file, dpi=fig.dpi)
        plt.close(fig)

    def as_df(self, metric_name):
        return pd.DataFrame(self.metrics[metric_name])