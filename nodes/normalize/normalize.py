#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brand import BRANDNode
from copy import deepcopy
import csv
import gc
import json
import logging
import numpy as np
import os
from redis import Redis
import sys
import time
import yaml

DEFAULT_INIT_SAMPS = 100


class Normalize(BRANDNode):
    def __init__(self):

        super().__init__()

        self.parameters.setdefault('clip_range', None)

        # stream parameters
        self.in_stream = self.parameters['input_stream'].encode()
        self.in_dtype = self.parameters['input_dtype']
        self.in_key = self.parameters['input_key'].encode()
        if 'store_norm_params' in self.parameters:
            self.store_norm_params = self.parameters['store_norm_params']
        else:
            self.store_norm_params = True
        self.output_stream = self.NAME+'_norm'
        self.denorm_stream = self.NAME+'_denorm'
        self.clip_range = self.parameters['clip_range']

        if self.clip_range:
            try:
                self.clip_min, self.clip_max = self.clip_range
            except Exception:
                logging.warning(f'Invalid clipping range: {self.clip_range}')
                self.clip_range = None

        # how many samples are required to write exit parameters
        self.samps_to_ow = self.parameters['samples_to_overwrite_exit_params'] if 'samples_to_overwrite_exit_params' in self.parameters else 0

        # normalization parameters
        if 'normalization_method' in self.parameters:
            self.norm_type = self.parameters['normalization_method']
        else:
            self.norm_type = 'z_window'

        logging.info(f'Reading from stream {self.in_stream}')

        # define timing and sync keys
        self.sync_key = self.parameters['sync_key'].encode()
        self.time_key = self.parameters['time_key'].encode()

        self.sample_dict = {self.in_stream: '$'}

        # initialize output stream entry data
        self.i = np.uint32(0)

        self.output_entry = {
            self.time_key: np.uint64(time.monotonic_ns()).tobytes(),
            self.sync_key: json.dumps({}),
            'samples': np.array([]).tobytes(),
            'i': self.i.tobytes()}

        self.build()

    def build(self):
        self.norm = norm_method_dict[self.norm_type](**{'r': self.r, **self.parameters})
        if self.norm.last_id is not None:
            self.sample_dict[self.in_stream] = self.norm.last_id

    def work(self):

        # get all new entries
        sample = self.r.xread(self.sample_dict, block=0)

        # for each entry
        for d in sample[0][1]:
            p = self.r.pipeline()

            time_ns = np.uint64(time.monotonic_ns()).tobytes()

            # normalize the data
            data = np.frombuffer(d[1][self.in_key], dtype=self.in_dtype)
            norm_data = self.norm.normalize_data(data)
            # clip outliers
            if self.clip_range:
                norm_data = np.clip(norm_data,
                                    a_min=self.clip_min,
                                    a_max=self.clip_max)
            self.output_entry['samples'] = norm_data.tobytes()
            self.output_entry[self.time_key] = time_ns
            self.output_entry[self.sync_key] = d[1][self.sync_key]
            self.output_entry['i'] = self.i.tobytes()
            p.xadd(self.output_stream, self.output_entry)

            if self.store_norm_params:
                norm_params = self.norm.get_norm_params()
                for s,v in norm_params.items():
                    self.output_entry['samples'] = v.tobytes()
                    p.xadd(self.NAME+'_'+s, self.output_entry)

            denorm_data = self.norm.denormalize_data(norm_data)
            if denorm_data is not None:
                self.output_entry['samples'] = denorm_data.tobytes()
                p.xadd(self.denorm_stream, self.output_entry)

            p.execute()

            self.i += 1

        self.sample_dict[self.in_stream] = sample[0][1][-1][0]

    def terminate(self, sig, frame):
        if hasattr(self, 'norm') and self.i >= self.samps_to_ow:
            norm_params = self.norm.get_exit_params()
            self.r.xadd(self.NAME+'_exit_parameters', {k:v.tobytes() for k,v in norm_params.items()})
            logging.info(f'Saved new normalization parameters ({self.i} samples)')
        else:
            logging.info(f'Not saving new normalization parameters ({self.i} samples)')
        return BRANDNode.terminate(self, sig, frame)

class NormalizationMethod():
    def __init__(self) -> None:
        """
        Base method to be overridden by the subclass.
        The Normalize class above requires a last_id
        attribute and a normalize_data class method.
        """
        self.last_id = None

    def normalize_data(self, data:np.ndarray) -> np.ndarray:
        """
        Base method to be overridden by the subclass.
        This normalizes a vector of data as desired.
        
        Parameters
        ----------
        data : numpy.ndarray
            The vector of incoming data to be normalized
        """
        return data

    def denormalize_data(self, data:np.ndarray) -> np.ndarray:
        """
        Base method to be overridden by the subclass.
        This denormalizes a vector by parameters
        optionally set during init.
        
        Parameters
        ----------
        data : numpy.ndarray
            The vector of data to be denormalized
        """
        return None

    def get_norm_params(self) -> dict:
        """
        Base method to be overridden by the subclass.
        This returns the normalization parameters at
        the time of function call in a dictionary.
        """
        return {}

    def get_exit_params(self) -> dict:
        """
        Base method to be overridden by the subclass.
        This should return returns the normalization
        parameters at exit for the purpose of loading
        them in the future.
        """
        return {}

class ZWindow(NormalizationMethod):
    def __init__(self, **kwargs):
        self.r:Redis = kwargs['r']
        self.stream_info = {
            'input_stream': kwargs['input_stream'].encode(),
            'input_key': kwargs['input_key'].encode(),
            'input_dtype': kwargs['input_dtype']}
        self.bin_period = int(kwargs['bin_period'])
        self.duration = int(kwargs['normalization_duration'] / self.bin_period)
        self.period = int(kwargs['normalization_update_period'] / self.bin_period)
        if 'norm_stream' in kwargs:
            self.norm_stream = kwargs['norm_stream']
        else:
            self.norm_stream = None
        if 'norm_file' in kwargs:
            self.norm_file = kwargs['norm_file']
        else:
            self.norm_file = None
        if 'denorm_stream' in kwargs:
            self.denorm_stream = kwargs['denorm_stream']
        else:
            self.denorm_stream = None
        if 'denorm_file' in kwargs:
            self.denorm_file = kwargs['denorm_file']
        else:
            self.denorm_file = None
        self.protect_neg_var = kwargs['protect_neg_var'] if 'protect_neg_var' in kwargs else True

        self.replace_0_std = kwargs.get('replace_0_std', 1)

        self.printed_using_window = False

        self.first_id = '$'

        self.build_norm()

        self.denorm_enable = False
        if self.denorm_stream is not None or self.denorm_file is not None:
            self.denorm_enable = True
            self.build_denorm()

    def build_denorm(self):
        denorm_means = None
        denorm_stds = None

        # try to load denorm parameters from stream
        if self.denorm_stream is not None:
            if not isinstance(self.denorm_stream, list):
                self.denorm_stream = [self.denorm_stream]
            
            # load in latest entry from each stream
            entries = {s: self.r.xrevrange(s, '+', '-', count=1) for s in self.denorm_stream}
            # delete empty entries
            entries = {k: v for k, v in entries.items() if v}
            # get redis IDs for each stream
            redis_ids = {s: entries[s][0][0] for s in entries}
            # sort by latest redis ID
            sorted_streams = sorted(redis_ids, key=lambda x: redis_ids[x], reverse=True)
            if sorted_streams:
                for s in sorted_streams:
                    entry = entries[s]
                    if b'means' in entry[0][1] and b'stds' in entry[0][1]:
                        denorm_means = np.frombuffer(entry[0][1][b'means'], dtype=np.float64)
                        denorm_stds = np.frombuffer(entry[0][1][b'stds'], dtype=np.float64)
                        logging.info(f'Loaded denormalization parameters from {s} stream')
                        break
                    else:
                        logging.warning(f'Could not find \'means\' and/or \'stds\' keys in {s} stream')
            else:
                logging.warning(f'No entries in {self.denorm_stream} stream(s), using defaults')

        # try to load denorm parameters from file
        if self.denorm_file is not None and (denorm_means is None or denorm_stds is None):
            try:
                with open(self.denorm_file, 'r') as f:
                    denorm_yaml = yaml.safe_load(f)
                if 'means' in denorm_yaml and 'stds' in denorm_yaml:
                    denorm_means = denorm_yaml['means']
                    denorm_stds = denorm_yaml['stds']
                    logging.info(f'Loaded denormalization parameters from {self.denorm_file}')
                else:
                    logging.warning(f'Could not find \'means\' and/or \'stds\' keys in {self.denorm_file}')

            except FileNotFoundError:
                logging.warning(f'Could not find denormalization parameters file at {self.denorm_file}')

        # defaults
        if denorm_means is None or denorm_stds is None:
            self.denorm_means = 0
            self.denorm_stds = 1
        elif len(denorm_means) != self.n_channels or len(denorm_stds) != self.n_channels:
            logging.warning(
                f'Number of denormalization parameters ({len(denorm_means)} means, {len(denorm_stds)} stds)'
                f' does not match the number of channels ({self.n_channels}), using defaults')
            self.denorm_means = 0
            self.denorm_stds = 1
        else:
            self.denorm_means = denorm_means
            self.denorm_stds = denorm_stds

    def build_norm(self):
        logging.info(f'Starting windowed z-score normalization across {self.duration*self.bin_period}ms, updated every {self.period*self.bin_period}ms ')

        # read in first sample
        entry = self.r.xread({self.stream_info['input_stream']: self.first_id}, block=0, count=1)
        self.n_channels = np.frombuffer(entry[0][1][0][1][self.stream_info['input_key']], dtype=self.stream_info['input_dtype']).shape[0] # get the number of channels from incoming data

        # store first_id so we repeatedly pull the same sample until we hit self.duration
        first_id = entry[0][1][0][0].decode('utf-8').split('-') # first index used in calculation
        self.first_id = str(int(first_id[0])-1)+'-'+str(0xFFFFFFFFFFFFFFFF)

        # calculate stats for data, initialize with first sample
        self.running_means = np.float64(np.frombuffer(entry[0][1][0][1][self.stream_info['input_key']], dtype=self.stream_info['input_dtype']))
        self.m_2 = np.float64(np.zeros(self.n_channels, dtype=self.stream_info['input_dtype']))
        self.n_samples_included = 1
        self.n_samples_processed = 1

        # initialize means and standard deviations
        self.means = deepcopy(self.running_means)
        self.first_means = deepcopy(self.means)
        self.stds = np.float64(np.zeros(self.n_channels, dtype=self.stream_info['input_dtype']))
        self.stds[self.stds <= 1e-8] = self.replace_0_std
        self.first_stds = deepcopy(self.stds)

        n_init_samps = DEFAULT_INIT_SAMPS if self.period == np.inf else self.period

        # if no initialization file
        if self.norm_stream is None and self.norm_file is None:
            logging.info(f'Initializing parameters from a data sample of length {n_init_samps} bins')

        # if reading initial means and standard deviations from a stream
        elif self.norm_stream is not None:
            logging.info(f'Initializing parameters from {self.norm_stream} stream')

            if not isinstance(self.norm_stream, list):
                self.norm_stream = [self.norm_stream]
            
            # load in latest entry from each stream
            entries = {s: self.r.xrevrange(s, '+', '-', count=1) for s in self.norm_stream}
            # delete empty entries
            entries = {k: v for k, v in entries.items() if v}
            # get redis IDs for each stream
            redis_ids = {s: entries[s][0][0] for s in entries}
            # sort by latest redis ID
            sorted_streams = sorted(redis_ids, key=lambda x: redis_ids[x], reverse=True)
            if sorted_streams:
                for s in sorted_streams:
                    entry_norm = entries[s]
                    if b'means' in entry_norm[0][1] and b'stds' in entry_norm[0][1]:
                        self.means = np.frombuffer(entry_norm[0][1][b'means'], dtype=np.float64)
                        self.first_means = deepcopy(self.means)
                        self.stds = np.frombuffer(entry_norm[0][1][b'stds'], dtype=np.float64).copy()
                        self.stds[self.stds <= 1e-8] = self.replace_0_std
                        self.first_stds = deepcopy(self.stds)
                        n_init_samps = 1
                        logging.info(f'Loaded normalization parameters from {s} stream')
                        break
                    else:
                        logging.warning(f'Could not find \'means\' and/or \'stds\' keys in {s} stream')
            else:
                logging.warning(f'No entries in {self.norm_stream} stream(s), computing new parameters')

        # if reading initial means and standard deviations from a file
        elif self.norm_file is not None:
            logging.info(f'Initializing parameters from {self.norm_file}')
            ext = os.path.splitext(self.norm_file)[-1].lower()
            if ext == '.csv':
                params = np.genfromtxt(self.norm_file, delimiter=',')
                self.means = params[0, :]
                self.first_means = deepcopy(self.means)
                self.stds = params[1, :]
                self.stds[self.stds <= 1e-8] = self.replace_0_std
                self.first_stds = deepcopy(self.stds)
                n_init_samps = 1

            if ext == '.yml' or ext == '.yaml':
                with open(self.norm_file, 'r') as f:
                    norm_file = yaml.safe_load(f)
                if 'means' in norm_file and 'stds' in norm_file:
                    self.means = norm_file['means']
                    self.first_means = deepcopy(self.means)
                    self.stds = norm_file['stds']
                    self.stds[self.stds <= 1e-8] = self.replace_0_std
                    self.first_stds = deepcopy(self.stds)
                else:
                    logging.error(f'Could not find \'means\' and/or \'stds\' keys in {self.norm_file}')
                    sys.exit(1)
                n_init_samps = 1

            else:
                logging.error('Unsupported normalization_file type, exiting')
                sys.exit(1)

        if self.n_channels != len(self.means) or self.n_channels != len(self.stds):
            n_init_samps = DEFAULT_INIT_SAMPS
            logging.warning(f'Number of channels (means: {len(self.means)}, stds: {len(self.stds)}) '
                            f'does not match the number of incoming channels ({self.n_channels}). '
                            f'Initializing parameters from a data sample of length {n_init_samps} bins')

        # update stats from rest of the data in first period
        self.last_id = entry[0][1][0][0]
        for _ in range(1, n_init_samps):
            entry = self.r.xread({self.stream_info['input_stream']: self.last_id}, block=0, count=1)[0][1][0]
            data = np.float64(np.frombuffer(entry[1][self.stream_info['input_key']], dtype=self.stream_info['input_dtype']))
            self.last_id = entry[0]
            self.n_samples_processed += 1
            if self.n_samples_included < self.duration-1:
                self.n_samples_included += 1
            else:
                # update entry ID
                self.first_id = self.last_id.decode('utf-8')
            self.update_statistics(None, data, self.n_samples_included)

    def update_statistics(self, first, last, n_samples):
        if self.n_samples_processed > self.duration:
            # https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
            new_means = self.running_means + (last - first)/n_samples
            new_m_2 = self.m_2 + (last-first)*(last-new_means+first-self.running_means)
        else:
            # Welford's online variance algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            new_means = self.running_means + (last - self.running_means)/n_samples
            new_m_2 = self.m_2 + (last-self.running_means)*(last-new_means)

        # check for negative variance
        if self.protect_neg_var:
            if np.any(new_m_2 < 0):
                neg_m2 = np.argwhere(new_m_2 < 0)
                logging.warning(f'Negative variance ignored:\n'
                                f'  channel(s): {neg_m2}\n'
                                f'  newest sample: {last[neg_m2]}\n'
                                f'  oldest sample: {first[neg_m2]}\n'
                                f'  new means: {new_means[neg_m2]}\n'
                                f'  running means: {self.running_means[neg_m2]}\n'
                                f'  new m_2: {new_m_2[neg_m2]}\n'
                                f'  running m_2: {self.m_2[neg_m2]}\n'
                                f'  num samples in stats: {n_samples}\n'
                                f'  num samples processed: {self.n_samples_processed}\n')
            else:
                self.m_2 = new_m_2
        else:
            self.m_2 = new_m_2
        self.running_means = new_means

        # update means and variances if it's time
        if self.n_samples_processed % self.period == 0:
            if self.n_samples_processed > self.duration:
                self.means = deepcopy(self.running_means)
                self.stds = np.sqrt(self.m_2 / (n_samples - 1))
            else:
                ratio = self.n_samples_processed / self.duration
                self.means = self.running_means * ratio + self.first_means * (1.0 - ratio)
                self.stds = np.sqrt((self.m_2 + (self.first_stds ** 2) * (self.duration - self.n_samples_processed - 1)) / (self.duration - 2))
            self.stds[self.stds <= 1e-8] = self.replace_0_std

    def normalize_data(self, data:np.ndarray) -> np.ndarray:
        first_sample = self.r.xrange(self.stream_info['input_stream'], '('+self.first_id, '+', count=1)
        first_data = np.float64(np.frombuffer(first_sample[0][1][self.stream_info['input_key']], dtype=self.stream_info['input_dtype']))

        self.n_samples_processed += 1
        if self.n_samples_included < self.duration:
            self.n_samples_included += 1
        else:
            if not self.printed_using_window:
                logging.info(f'Using windowed z-score normalization')
                self.printed_using_window = True
            # update entry ID
            self.first_id = first_sample[0][0].decode('utf-8')

        # update the statistics with the new data
        self.update_statistics(first_data, data, self.n_samples_included)

        # normalize the data by the existing parameters
        return (data - self.means) / self.stds

    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        return data * self.denorm_stds + self.denorm_means if self.denorm_enable else None

    def get_norm_params(self) -> dict:
        return {'means': self.means, 'stds': self.stds}
    
    def get_exit_params(self) -> dict:
        stds = np.sqrt(self.m_2 / (self.n_samples_included - 1))
        stds[stds <= 1e-8] = self.replace_0_std
        return {'means': self.running_means, 'stds': stds}

norm_method_dict = {
    'z_window': ZWindow
}


if __name__ == "__main__":
    gc.disable()

    # setup
    norm = Normalize()

    # main
    norm.run()

    gc.collect()