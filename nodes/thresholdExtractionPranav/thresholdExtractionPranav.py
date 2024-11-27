#!/usr/bin/env python
# -*- coding: utf-8 -*-
# threshold_extraction.py
# Kevin Bodkin, Yahia Ali

from copy import deepcopy
import gc
import json
import logging
import signal
import sys
import time
import yaml

import numpy as np
import scipy.signal
from brand import BRANDNode
from brand.redis import xread_count

import os
from datetime import datetime
from collections import defaultdict
import fcntl

class TimingProfiler:
    def __init__(self):
        self.timings = defaultdict(list)

    def set_name(self, name):
        self.name =name 

    def record(self, operation, duration):
        self.timings[operation].append(duration)
    
    def get_stats(self):
        stats = {}
        for op, times in self.timings.items():
            stats[op] = {
                'mean': np.mean(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        return stats
    
    def print_stats(self):
        print(f"\nNode Name:{self.name}")
        stats = self.get_stats()
        print("\nTiming Statistics (in milliseconds):")
        print("-" * 80)
        print(f"{'Operation':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>10}")
        print("-" * 80)
        for op, metrics in stats.items():
            print(f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}")

    def save_staats(self, directory):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"timing_stats_{self.name}_{timestamp}.log")
        
        stats = self.get_stats()
        with open(filename, 'w') as f:
            f.write(f"Node Name: {self.name}\n")
            f.write("\nTiming Statistics (in milliseconds):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Operation':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>10}\n")
            f.write("-" * 80 + "\n")
            for op, metrics in stats.items():
                f.write(f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}\n")



    def save_stats(self, directory, filename='timing_stats.log'):
        """
        Save timing statistics to a file with file locking to handle concurrent writes.
        
        Args:
            directory (str): Directory to save the log file
            filename (str, optional): Name of the log file. Defaults to 'timing_stats.log'
        """
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Full path to the log file
        filepath = os.path.join(directory, filename)
        
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare the log entry
        log_entry = f"\n--- {timestamp} - Node: {self.name} ---\n"
        log_entry += "-" * 80 + "\n"
        log_entry += f"{'Operation':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>10}\n"
        log_entry += "-" * 80 + "\n"
        
        # Get stats
        stats = self.get_stats()
        
        # Add formatted stats to log entry
        for op, metrics in stats.items():
            log_entry += f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}\n"
        
        # Maximum wait time (in seconds)
        max_wait_time = 30
        wait_interval = 0.1
        total_wait_time = 0
        
        while total_wait_time < max_wait_time:
            try:
                # Open the file with exclusive lock
                with open(filepath, 'a') as f:
                    # Acquire an exclusive lock (blocks until lock is available)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Write the log entry
                        f.write(log_entry)
                        f.flush()  # Ensure data is written to disk
                        break  # Successfully wrote, exit the loop
                    
                    finally:
                        # Release the lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            except IOError:
                # If file is busy, wait and retry
                time.sleep(wait_interval)
                total_wait_time += wait_interval
        
        else:
            # If we've exceeded max wait time
            raise IOError(f"Could not acquire lock on {filepath} after {max_wait_time} seconds")

class ThresholdExtraction(BRANDNode):

    def __init__(self, parameters=None):
        super().__init__()
        self.profiler = TimingProfiler()

        if parameters:
            self.parameters =parameters
        
        self.profiler.set_name(self.NAME)
        # threshold multiplier, usually around -5
        self.thresh_mult = self.parameters['thresh_mult']
        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters['pack_per_call']
        # whether to export the filtered data
        self.output_filtered = self.parameters['output_filtered']
        # enable acausal filtering
        self.acausal_filter = self.parameters['acausal_filter']
        self.causal = not self.acausal_filter
        # length of the buffer used for acausal filtering
        self.acausal_filter_lag = self.parameters['acausal_filter_lag']
        # whether to conduct common-average referencing
        self.demean = self.parameters['enable_CAR']

        # parameters of the input stream
        self.input_stream = self.parameters['input_name']
        # number of samples per channel per redis entry
        self.samp_per_stream = self.parameters['input_samp_per_stream']
        # number of channels
        self.n_channels_total = self.parameters['input_chan_per_stream']

        # which channels to use
        if 'neural_ch_range' in self.parameters:
            if len(self.parameters['neural_ch_range']) == 2:
                self.n_range = np.arange(
                    self.parameters['neural_ch_range'][0],
                    self.parameters['neural_ch_range'][1])
            else:
                logging.warning(
                    '\'neural_ch_range\' parameter should be length 2,'
                    ' attempting to use all neural channels')
                self.n_range = np.arange(0, self.n_channels_total)
        else:
            self.n_range = np.arange(0, self.n_channels_total)                               
        self.n_range = self.n_range.astype(int)
        self.n_channels = self.n_range.shape[0] 

        # whether to remove coincident spikes
        self.num_coincident = self.parameters['num_coincident_spikes'] if 'num_coincident_spikes' in self.parameters else None
        self.num_coincident = self.n_channels+1 if self.num_coincident is None else self.num_coincident

        # thresholds stream
        if 'thresholds_stream' in self.parameters:
            self.thresholds_stream = self.parameters['thresholds_stream']
        else:
            self.thresholds_stream = None

        # thresholds file
        if 'thresholds_file' in self.parameters:
            self.thresholds_file = self.parameters['thresholds_file']
        else:
            self.thresholds_file = None

        # which thresholds to use
        if (self.thresholds_stream is not None
                or self.thresholds_file is not None):
            if 'thresholds_ch_range' in self.parameters:
                if len(self.parameters['thresholds_ch_range']) == 2:
                    self.th_chans = range(
                        self.parameters['thresholds_ch_range'][0],
                        self.parameters['thresholds_ch_range'][1])
                else:
                    logging.warning(
                        '\'thresholds_ch_range\' parameter should be length 2,'
                        ' attempting to use all channels in the thresholds '
                        'stream or file')
                    self.th_chans = None
            else:
                logging.warning(
                    '\'thresholds_ch_range\' was not provided, attempting to '
                    'use all channels in the thresholds stream or file')
                self.th_chans = None

        # optional datatype
        if 'input_data_type' in self.parameters:
            self.dtype = self.parameters['input_data_type']
        else:
            self.dtype = np.int16

        # optional datatype
        if 'timestamp_data_type' in self.parameters:
            self.tdtype = self.parameters['timestamp_data_type']
        else:
            self.tdtype = np.uint32
        
        # use tracking id instead of NSP timestamp
        if 'use_tracking_id' in self.parameters:
            self.use_tracking_id = self.parameters['use_tracking_id']
        else:
            self.use_tracking_id = False

        # list of lists of common-average reference groupings
        if self.demean and 'CAR_group_sizes' in self.parameters:
            car_sizes = self.parameters['CAR_group_sizes']
            if not isinstance(car_sizes, list):
                if isinstance(car_sizes, int):
                    car_sizes = []
                    ch_count = deepcopy(self.n_channels)

                    # get CAR group sizes of the specified size, until we run
                    # out of channels for the stream
                    while ch_count > 0:
                        car_sizes.append(
                            min([self.parameters['CAR_group_sizes'],
                                 ch_count]))
                        ch_count -= self.parameters['CAR_group_sizes']

            self.car_groups = []
            ch_count = 0
            for g in car_sizes:
                if not isinstance(g, int):
                    raise ValueError(
                        '\'CAR_group_sizes\' must be a list of \'int\'s or a '
                        'single \'int\', but '
                        f'{self.parameters["CAR_group_sizes"]} was given')
                self.car_groups.append(
                    np.arange(ch_count, ch_count + g).tolist())
                ch_count += g
        else:
            self.car_groups = [np.arange(0, self.n_channels).tolist()]

        # exclude channels
        if 'exclude_channels' in self.parameters:
            exclude_ch = self.parameters['exclude_channels']
            if not isinstance(exclude_ch, list):
                if isinstance(exclude_ch, int):
                    exclude_ch = [exclude_ch]
            for c in exclude_ch:
                if not isinstance(c, int):
                    raise ValueError(
                        '\'exclude_channels\' must be a list of \'int\'s or'
                        ' a single \'int\', but '
                        f'{self.parameters["exclude_channels"]} was given.'
                        ' Exiting')
            for c in exclude_ch:
                for g in self.car_groups:
                    if c in g:
                        g.remove(c)

        # stream containing the list of channels to use
        if 'ch_mask_stream' in self.parameters:
            ch_mask_entry = self.r.xrevrange(self.parameters['ch_mask_stream'], '+', '-', count=1)
            if ch_mask_entry:
                self.ch_mask = np.frombuffer(ch_mask_entry[0][1][b'channels'], dtype=np.uint16)
                if self.th_chans is None:
                    logging.warning(f'\'ch_mask_stream\' was provided but \'thresholds_ch_range\' was not, so the incorrect channels may be masked')
                else:
                    self.ch_mask = np.array(list(set(self.ch_mask).intersection(set(self.th_chans))))
                    self.ch_mask -= self.th_chans[0]
            else:
                logging.warning(f'\'ch_mask_stream\' was set to {self.parameters["ch_mask_stream"]}, but there were no entries. Defaulting to using all channels')
                self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)
        else:
            self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)

        # keep only masked channels
        for g_idx in range(len(self.car_groups)):
            self.car_groups[g_idx] = list(set(self.car_groups[g_idx]).intersection(set(self.ch_mask)))

        # initialize adaptive threholding
        if 'adaptive_thresholds' in self.parameters:
            self.adaptive_thresholds = self.parameters['adaptive_thresholds']
        else:
            self.adaptive_thresholds = False
        if self.adaptive_thresholds:
            self.rms_window_len = self.parameters['adaptive_rms_window_len']
            self.adaptive_rms_stream = self.parameters['adaptive_rms_stream']
            self.mean_squared_buffer = np.zeros((self.n_channels, self.rms_window_len), dtype=np.float64)
            self.mean_squared_buffer_index = 0
            self.mean_squared_buffer_full = False
            self.mean_squared_last = np.zeros((self.n_channels), dtype=np.float64)
            self.mean_squared_new = np.zeros((self.n_channels), dtype=np.float64)
            self.root_mean_squared = np.zeros((self.n_channels), dtype=np.float64)
            logging.info(f"Adaptive spike thresholds enabled, using RMS computed over {self.samp_per_stream*self.rms_window_len} 30kHz samples")

        # define timing and sync keys
        if 'sync_key' in self.parameters:
            self.sync_key = self.parameters['sync_key'].encode()
        else:
            self.sync_key = b'sync'
        if 'time_key' in self.parameters:
            self.time_key = self.parameters['time_key'].encode()
        else:
            self.time_key = b'ts'
        if 'sync_source_id' in self.parameters:
            self.sync_source_id = self.parameters['sync_source_id']
        else:
            self.sync_source_id = 'i'

        # build filtering pipeline
        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = self.build_filter()

        # load or compute thresholds
        self.thresholds = None
        if self.thresholds_stream is not None:
            self.thresholds = self.load_thresholds_from_stream(
                self.thresholds_stream, self.th_chans)

        if self.thresholds is None and self.thresholds_file is not None:
            self.thresholds = self.load_thresholds_from_file(
                self.thresholds_file, self.th_chans)

        if self.thresholds is None:
            # amount of data to use for threshold calculation
            self.thresh_calc_len = self.parameters['thresh_calc_len']
            self.thresholds = self.calc_thresh(
                self.input_stream, self.thresh_mult, self.thresh_calc_len,
                self.samp_per_stream, self.n_channels, self.sos, self.zi)
        

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

    def build_filter(self):
        # order of the butterworth filter
        but_order = self.parameters['butter_order']
        # lower cutoff frequency
        but_low = self.parameters['butter_lowercut']
        # upper cutoff frequency
        but_high = self.parameters['butter_uppercut']
        # whether to use an FIR filter on the reverse-pass
        acausal_filter = self.acausal_filter
        # enable causal filtering
        causal = self.causal
        # sampling rate of the input
        fs = self.parameters['input_samp_freq']

        # determine filter type
        if but_low and but_high:
            filt_type = 'bandpass'
            Wn = [but_low, but_high]
        elif but_high:
            filt_type = 'lowpass'
            Wn = but_high
        elif but_low:
            filt_type = 'highpass'
            Wn = but_low
        else:
            raise ValueError(
                "Must specify 'butter_lowercut' or 'butter_uppercut'")

        # set up filter
        sos = scipy.signal.butter(but_order,
                                  Wn,
                                  btype=filt_type,
                                  analog=False,
                                  output='sos',
                                  fs=fs)  # set up a filter
        # initialize the state of the filter
        zi_flat = scipy.signal.sosfilt_zi(sos)
        # so that we have the right number of dimensions
        zi = np.zeros((zi_flat.shape[0], self.n_channels, zi_flat.shape[1]))
        # filter initialization
        for ii in range(self.n_channels):
            zi[:, ii, :] = zi_flat

        # select the filtering function
        if acausal_filter and acausal_filter.lower() == 'fir':
            use_fir = True
        else:
            use_fir = False
        filter_func = get_filter_func(self.demean, causal, use_fir=use_fir)

        # log the filter info
        causal_str = 'causal' if causal else 'acausal'
        message = (f'Loading {but_order :d} order, '
                   f'{Wn} hz {filt_type} {causal_str}')
        if causal:
            message += ' IIR filter'
        elif use_fir:
            message += ' IIR-FIR filter'
        else:
            message += ' IIR-IIR filter'
        message += ' with CAR' if self.demean else ''
        logging.info(message)

        if not causal:
            if use_fir:
                # FIR filter (backward)
                N = self.acausal_filter_lag + 1  # length of the filter
                imp = scipy.signal.unit_impulse(N)
                rev_win = scipy.signal.sosfilt(sos, imp)
                # filter initialization
                rev_zi_flat = scipy.signal.lfilter_zi(rev_win, 1.0)
                rev_zi = np.zeros((self.n_channels, rev_zi_flat.shape[0]))
                for ii in range(self.n_channels):
                    rev_zi[ii, :] = rev_zi_flat
            else:
                rev_win = None
                rev_zi = zi.copy()

        if causal:
            return filter_func, sos, zi
        else:
            return filter_func, sos, zi, rev_win, rev_zi

    def calc_thresh(self, stream, thresh_mult, thresh_cal_len, samp_per_stream,
                    n_channels, sos, zi):
        reply = xread_count(self.r,
                            stream=stream,
                            startid='$',
                            count=thresh_cal_len,
                            block=0)

        _, entries = reply[0]  # get the list of entries

        read_arr = np.empty((n_channels, thresh_cal_len * samp_per_stream),
                            dtype=np.float32)
        filt_arr = np.empty((n_channels, thresh_cal_len * samp_per_stream),
                            dtype=np.float32)
        # read_times = np.empty((thresh_cal_len * samp_per_stream))

        i_start = 0
        for _, entry_data in entries:  # put it all into an array
            i_end = i_start + samp_per_stream
            read_arr[:, i_start:i_end] = np.reshape(
                np.frombuffer(entry_data[b'samples'], dtype=self.dtype),
                (self.n_channels_total, samp_per_stream))[self.n_range,:]
            # read_times[i_start:i_end] = np.frombuffer(
            #     entry_data[b'timestamps'], self.tdtype)
            i_start = i_end

        if self.causal:
            self.filter_func(read_arr, filt_arr, sos, zi)
        else:
            if self.demean:
                common_average_reference(read_arr, self.car_groups)
            filt_arr[:, :] = scipy.signal.sosfiltfilt(sos, read_arr, axis=1)

        thresholds = (thresh_mult *
                      np.sqrt(np.mean(np.square(filt_arr), axis=1))).reshape(
                          -1, 1)

        # log thresholds to database
        thresolds_enc = thresholds.astype(np.int16).tobytes()
        self.r.xadd(f'{self.NAME}_thresholds', {b'thresholds': thresolds_enc})

        logging.info('Calculated and set thresholds')
        return thresholds

    def load_thresholds_from_file(self, thresholds_file, tf_chans):
        try:
            with open(thresholds_file, 'r') as f:
                thresh_yaml = yaml.safe_load(f)
            if 'thresholds' in thresh_yaml:
                if tf_chans is None:
                    if len(thresh_yaml['thresholds']) == self.n_channels_total:
                        logging.info(
                            f'Loaded thresholds from {thresholds_file}')
                        return np.array(thresh_yaml['thresholds']).reshape(
                            -1, 1)
                    else:
                        raise ValueError(
                            f'Number of thresholds in {thresholds_file} '
                            f'({len(thresh_yaml["thresholds"])}) does not '
                            f'equal n_channels parameter {(self.n_channels_total)}')
                # if all of our requested channels are in the available range
                # of channels
                elif (set(tf_chans)
                      & set(range(0, len(
                          thresh_yaml['thresholds'])))) == set(tf_chans):
                    logging.info(f'Loaded thresholds from {thresholds_file}')
                    return np.array(
                        thresh_yaml['thresholds'])[tf_chans].reshape(-1, 1)
                else:
                    raise ValueError(
                        f'Channel range '
                        f'{self.parameters["thresholds_ch_range"]} outside of '
                        f'available channels in {thresholds_file} '
                        f'(max {len(thresh_yaml["thresholds"])})')
            else:
                logging.warning(
                    f'Could not find \'thresholds\' key in {thresholds_file}')
                return None

        except FileNotFoundError:
            logging.warning(
                f'Could not find thresholds file at {thresholds_file}')
            return None

    def load_thresholds_from_stream(self, stream, th_chans):
        entry = self.r.xrevrange(stream, '+', '-', count=1)
        if entry:
            thresholds = np.frombuffer(entry[0][1][b'thresholds'],
                                       dtype=np.float64)
            if th_chans is None:
                if len(thresholds) == self.n_channels_total:
                    logging.info(f'Loaded thresholds from the {stream} stream')
                    return thresholds.reshape(-1, 1)
                else:
                    raise ValueError(
                        f'Number of thresholds in the {stream} stream '
                        f'({len(thresholds)}) does not equal n_channels '
                        f'parameter {(self.n_channels_total)}')
            # if all of our requested channels are in the available range of
            # channels
            elif (set(th_chans)
                  & set(range(0, len(thresholds)))) == set(th_chans):
                logging.info(f'Loaded thresholds from the {stream} stream')
                return thresholds[th_chans].reshape(-1, 1)
            else:
                raise ValueError(
                    f'Channel range {self.parameters["thresholds_ch_range"]} '
                    f'outside of available channels in {stream} stream '
                    f'(max {len(thresholds)})')
        else:
            logging.warning(f'{stream} stream has no entries')
            return None


    def run(self):
        # get class variables
        
        if not self.causal:
            rev_win = self.rev_win
            rev_zi = self.rev_zi
        input_stream = self.input_stream
        output_filtered = self.output_filtered
        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        sos = self.sos
        thresholds = self.thresholds
        zi = self.zi

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
        data_buffer = np.zeros((self.n_channels, n_samp), dtype=np.float32)
        filt_buffer = np.zeros_like(data_buffer)
        rev_buffer = np.zeros(
            (self.n_channels, self.acausal_filter_lag + n_samp),
            dtype=np.float32)
        crossings = np.zeros_like(data_buffer)
        samp_times = np.zeros(n_samp, dtype=self.tdtype)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=self.tdtype)
        buffer_fill = 0

        # initialize stream entries
        cross_dict = {}
        filt_dict = {}
        rms_dict = {}
        coinc_dict = {}
        sync_dict = {self.sync_source_id: int(samp_times[0])}

        # initialize xread stream dictionary
        input_stream_dict = {input_stream: '$'}

        # set timeout
        timeout = 500

        # name the filtered output stream
        filt_stream = f'{self.NAME}_filt'

        try:
            while True:
                # Profile Redis read

                start_time = time.perf_counter()

                xread_receive = self.r.xread(input_stream_dict,
                                            block=timeout,
                                            count=pack_per_call)
                redis_rd_time = time.perf_counter()
                

                if len(xread_receive) >= pack_per_call:
                    # Profile data parsing
                    indStart = 0
                    for entry_id, entry_data in xread_receive[0][1]:
                        indEnd = indStart + samp_per_stream
                        # logging.info(f"self.n_channels, n_samp={self.n_channels, n_samp}  \
                        #               self.n_channels_total, samp_per_stream={self.n_channels_total, samp_per_stream}  \
                        #               self.n_range={self.n_range} ")
                        data_buffer[:, indStart:indEnd] =np.reshape(
                            np.frombuffer(entry_data[b'samples'],
                                        dtype=self.dtype),
                            (self.n_channels_total, samp_per_stream))[self.n_range,:]
                        if self.use_tracking_id:
                            samp_times[indStart:indEnd] = np.repeat(np.frombuffer(
                                entry_data[b'tracking_id'], self.tdtype), samp_per_stream)
                        else:
                            samp_times[indStart:indEnd] = np.frombuffer(
                                entry_data[b'timestamps'], self.tdtype)
                        indStart = indEnd
                    input_stream_dict[input_stream] = entry_id
                    data_parsing_time= time.perf_counter() 

                    if self.causal:
                        self.filter_func(data_buffer, filt_buffer, sos, zi,
                                        self.car_groups)
                    else:
                        self.filter_func(data_buffer,
                                        filt_buffer,
                                        rev_buffer,
                                        sos=sos,
                                        zi=zi,
                                        group_list=self.car_groups,
                                        rev_win=rev_win,
                                        rev_zi=rev_zi)
                    
                    filtering_time = time.perf_counter()

                    if not self.causal:
                        samp_times_buffer[:-n_samp] = (samp_times_buffer[n_samp:])
                        samp_times_buffer[-n_samp:] = samp_times


                    if self.causal:
                        samp_time_current = samp_times[:n_samp]
                    else:
                        samp_time_current = samp_times_buffer[:n_samp]
                        if buffer_fill + n_samp < buffer_len:
                            buffer_fill += n_samp
                            continue
                    
                    buffering_time =time.perf_counter()
                    # Profile adaptive thresholding
                    if self.adaptive_thresholds:
                        # t0 = time.perf_counter()
                        self.mean_squared_new = np.mean(filt_buffer**2, axis=1)
                        self.mean_squared_last += (self.mean_squared_new - self.mean_squared_buffer[:,self.mean_squared_buffer_index])/self.rms_window_len
                        self.mean_squared_buffer[:,self.mean_squared_buffer_index] = self.mean_squared_new
                        self.mean_squared_buffer_index += 1
                        if self.mean_squared_buffer_index >= self.rms_window_len:
                            self.mean_squared_buffer_index = 0
                            self.mean_squared_buffer_full = True
                        self.root_mean_squared = np.sqrt(self.mean_squared_last)
                        if self.mean_squared_buffer_full:
                            self.thresholds = (self.thresh_mult * self.root_mean_squared).reshape(-1,1)
                            thresholds = self.thresholds
                        # self.profiler.record('Adaptive thresholding', time.perf_counter() - t0)

                    # Profile threshold crossing detection

                    sync_dict[self.sync_source_id] = int(samp_time_current[0])
                    cross_dict[self.sync_key] = json.dumps(sync_dict)
                    cross_dict[b'timestamps'] = samp_time_current[0].tobytes()
                    crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                        (filt_buffer[:, :-1] >= thresholds))
                    cross_now = np.any(crossings, axis=1).astype(np.int16)
                    thresholding_time = time.perf_counter()


                    p = self.r.pipeline()
                    time_now = np.uint64(time.monotonic_ns()).tobytes()

                    tot_spikes = cross_now.sum()
                    if tot_spikes >= self.num_coincident:
                        coinc_dict[self.sync_key] = json.dumps(sync_dict)
                        coinc_dict[b'timestamps'] = samp_time_current[0].tobytes()
                        coinc_dict[self.time_key] = time_now
                        coinc_dict[b'crossings'] = cross_now.tobytes()
                        p.xadd(f'{self.NAME}_coinc', coinc_dict)
                        cross_now[:] = 0
                        
                    cross_dict[b'crossings'] = cross_now.tobytes()
                    cross_dict[self.time_key] = time_now
                    p.xadd(self.NAME, cross_dict)

                    if output_filtered:
                        filt_dict[self.sync_key] = json.dumps(sync_dict)
                        filt_dict[b'timestamps'] = samp_time_current.tobytes()
                        filt_dict[b'samples'] = filt_buffer.astype(
                            np.int16).tobytes()
                        filt_dict[self.time_key] = time_now
                        p.xadd(filt_stream, filt_dict)

                    if self.adaptive_thresholds:
                        rms_dict[self.sync_key] = json.dumps(sync_dict)  
                        rms_dict[b'timestamps'] = samp_time_current[0].tobytes()    
                        rms_dict[b'samples'] = self.root_mean_squared.astype(
                            np.float64).tobytes()  
                        rms_dict[b'thresholds'] = thresholds.astype(np.float64).tobytes() 
                        rms_dict[self.time_key] = time_now   
                        p.xadd(self.adaptive_rms_stream, rms_dict)             

                    p.execute()

                    redis_wt_time = time.perf_counter()

                    if buffer_fill + n_samp >= buffer_len:
                        self.profiler.record("Redis read", redis_rd_time-start_time)
                        self.profiler.record("Data parsing", data_parsing_time-redis_rd_time)
                        self.profiler.record("Filtering data", filtering_time-data_parsing_time)
                        self.profiler.record("Buffer updates", buffering_time- filtering_time)
                        self.profiler.record("Threshold detection", thresholding_time-buffering_time)
                        self.profiler.record("Redis write", redis_wt_time-thresholding_time)
                        self.profiler.record("Full time", redis_wt_time-start_time)
                        self.profiler.record("Execution time",thresholding_time-redis_rd_time)


                elif len(xread_receive) == 0:
                    logging.warning(f"No neural data received in {timeout} ms")

        except KeyboardInterrupt:
            # Print timing statistics on exit
            self.profiler.print_stats()
            raise


    def terminate(self, sig, frame):
        logging.info('SIGINT received, Exiting')
        log_dir = "/home/pdeevi/Projects/emory-cart/brand-modules/brand-nsp/nodes/thresholdExtractionPranav/logs"
        self.profiler.save_stats(log_dir)

        self.cleanup()
        self.r.close()
        gc.collect()
        sys.exit(0)
    # def terminate(self, sig, frame):
    #     if (hasattr(self, 'adaptive_thresholds') and 
    #             hasattr(self, 'thresholds') and 
    #             hasattr(self, 'thresholds_stream')):
    #         if (self.adaptive_thresholds and 
    #                 self.thresholds is not None and 
    #                 self.thresholds_stream is not None):
    #             self.r.xadd(self.thresholds_stream, {'thresholds': self.thresholds.tobytes()})
    #             logging.info(f'Logging latest adaptive thresholds to stream: \'{self.thresholds_stream}\'')
    #     return BRANDNode.terminate(self, sig, frame)

# Filtering functions
def get_filter_func(demean, causal=False, use_fir=True):
    """
    Get a function for filtering the data

    Parameters
    ----------
    demean : bool
        Whether to apply a common average reference before filtering
    causal : bool
        Whether to use causal filtering or acausal filtering
    use_fir : bool
        Whether to use an FIR filter for the reverse filter (when causal=False)
    """

    def causal_filter(data, filt_data, sos, zi, group_list):
        """
        causal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        group_list : list
            List of lists of channels grouped together across
            which to compute a common average reference
        """
        if demean:
            common_average_reference(data, group_list)
        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       rev_buffer,
                       sos,
                       zi,
                       group_list,
                       rev_win=None,
                       rev_zi=None):
        """
        acausal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        rev_buffer : ndarray
            Array to store the output of the forward IIR filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        group_list : list
            List of lists of channels grouped together across
            which to compute a common average reference
        rev_win : array-like, optional
            Coefficients of the reverse FIR filter
        rev_zi : array-like, optional
            Steady-state conditions of the reverse filter
        """
        if demean:
            common_average_reference(data, group_list)

        # shift the buffer
        n_samp = data.shape[1]
        rev_buffer[:, :-n_samp] = rev_buffer[:, n_samp:]

        # run the forward pass filter
        rev_buffer[:, -n_samp:], zi[:, :] = scipy.signal.sosfilt(sos,
                                                                 data,
                                                                 axis=1,
                                                                 zi=zi)
        # run the backward pass filter
        # 1. pass in the reversed buffer
        # 2. get the last N samples of the filter output
        # 3. reverse the output when saving to filt_data
        if use_fir:
            for ii in range(filt_data.shape[0]):
                filt_data[ii, ::-1] = np.convolve(rev_buffer[ii, ::-1],
                                                  rev_win, 'valid')
        else:
            ic = rev_zi * filt_data[:, -1][None, :, None]
            filt_data[:, ::-1] = scipy.signal.sosfilt(sos,
                                                      rev_buffer[:, ::-1],
                                                      axis=1,
                                                      zi=ic)[0][:, -n_samp:]

    filter_func = causal_filter if causal else acausal_filter

    return filter_func


def common_average_reference(data, group_list):
    """
    common average reference by group

    Parameters
    ----------
    data : array_like
        An 2-dimensional input array with shape
        [channel x time]
    group_list : list
        List of lists of channels grouped together across
        which to compute a common average reference
    """
    for g in group_list:
        data[g, :] -= data[g, :].mean(axis=0, keepdims=True)


if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    thresh_ext = ThresholdExtraction()
    thresh_ext.run()

    gc.collect()  # run garbage collection
