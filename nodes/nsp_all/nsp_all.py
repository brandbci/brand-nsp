import gc
import json
import logging
import signal
import sys
import time
from pathlib import Path
from glob import glob

import numpy as np
import os
from brand import BRANDNode
from collections import defaultdict

from helper import *

class TimingProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        
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
        stats = self.get_stats()
        print("\nTiming Statistics (in milliseconds):")
        print("-" * 80)
        print(f"{'Operation':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>10}")
        print("-" * 80)
        for op, metrics in stats.items():
            print(f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}")



class NSP_all(BRANDNode):
    def __init__(self, parameters):
        super().__init__()
        self.profiler = TimingProfiler()
        t_init = time.perf_counter()
        if parameters:
            self.parameters =parameters
        # Initialise all parameters
        self.parameter_initialization()

        # Variables for maintaining the latest state (supergraph).
        self.current_supergraph_dict = {}
        self.current_supergraph_id = '0-0'

        self.profiler.record('INIT', time.perf_counter() - t_init)

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

        # Set flag to indicate that we are  using the correct electrode mapping for this data
        self.r.set('using_correct_electrode_mapping', 1)

    def parameter_initialization(self):
        self.input_stream_name = self.parameters['input_stream_name']
        self.output_stream_name = self.parameters['output_stream_name']
        self.coefs_stream_name = self.parameters['coefs_stream_name']
        if 'input_dtype' in self.parameters:
            self.input_dtype = self.parameters['input_dtype'] 
        else:
            self.input_dtype = np.int16
        if 'output_dtype' in self.parameters:
            self.output_dtype = self.parameters['output_dtype'] 
        else:
            self.output_dtype = np.float64
        # self.samp_freq = self.parameters['samp_freq']
        self.chan_per_stream = self.parameters['chan_per_stream']
        self.samp_per_stream = self.parameters['samp_per_stream']
        self.chan_total = self.parameters['chan_total']
        self.start_channel = self.parameters['start_channel']   
        if 'neural_data_field' in self.parameters:
            self.neural_data_field = self.parameters['neural_data_field'] 
        else:
            self.neural_data_field = 'samples'
        if 'ts_field' in self.parameters:
            self.ts_field = self.parameters['ts_field'] 
        else:
            self.ts_field = 'BRANDS_time'
        
        self.nsp_ts_field = self.parameters.setdefault('nsp_ts_field', 'timestamps').encode()
        self.nsp_ts_dtype = self.parameters.setdefault('nsp_ts_dtype', 'uint64')

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

        self.neural_data = np.zeros((self.chan_per_stream, self.samp_per_stream), dtype=self.output_dtype)
        self.neural_data_reref = np.zeros((self.chan_per_stream, self.samp_per_stream), dtype=self.output_dtype)
        self.coefs = np.zeros((self.chan_per_stream, self.chan_per_stream), dtype=self.output_dtype)

        if 'n_split' in self.parameters:
            self.n_split = self.parameters['n_split'] 
        else:
            self.n_split = 16

        self.initialize_coefficients()
        self.init_filter_parameters()

        self.input_id = '$'
        self.output_dict = {}

        self.last_ts = np.array([0])

        logging.info(f"Parameters loaded. Reading from stream: {self.input_stream_name}. Writing to stream: {self.output_stream_name}")



    def initialize_coefficients(self):

        entries = self.r.xrevrange(self.coefs_stream_name, '+', '-', count=1)

        if len(entries):
            logging.info(f"Coefficients entry found in Redis stream: {self.coefs_stream_name}")

            entry_id, entry_dict = entries[0] 

            self.coefs_all = np.frombuffer(entry_dict[b'channel_scaling'], dtype=np.float64).reshape((self.chan_total, self.chan_total))
            self.coefs = self.coefs_all[self.start_channel:self.start_channel+self.chan_per_stream, 
                                        self.start_channel:self.start_channel+self.chan_per_stream]

            if b'channel_unshuffling' in entry_dict:
                self.unshuffle_all = np.frombuffer(entry_dict[b'channel_unshuffling'], dtype=np.float64).reshape((self.chan_total, self.chan_total))
                logging.info(f"Unshuffling matrix loaded from stream.")
            else:
                self.unshuffle_all = np.eye(self.chan_total, dtype=np.float64)
                logging.info(f"No unshuffling matrix found. Assuming channels are in order.")
            
            self.unshuffle = self.unshuffle_all[self.start_channel:self.start_channel+self.chan_per_stream, 
                                                self.start_channel:self.start_channel+self.chan_per_stream]

            self.coefs = (np.eye(self.chan_per_stream) - self.coefs) @ self.unshuffle
            self.coefs = self.coefs.astype(self.output_dtype)

        else:
            logging.info(f"Coefficients entry not found in Redis stream: {self.coefs_stream_name}. Setting coefficients to compute mean of all channels.")

            self.coefs = np.eye(self.chan_per_stream) - np.ones((self.chan_per_stream, self.chan_per_stream))/self.chan_per_stream
            self.coefs.astype(self.parameters['output_dtype'])

    def init_helper_functions(self):
        # build filtering pipeline
        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = self.build_filter()



        self.filter  = helper.get_filt_func()

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




    def run(self):


        while True:

            ###################################### READ FROM REDIS ######################################
            t0 = time.perf_counter()
            
            
            self.profiler.record('Redis read', time.perf_counter() - t0)


            ###################################### RE-REFERENCING ######################################
            t0 = time.perf_counter()

            reref_data = self.rereference_func(raw_data)

            self.profiler.record('Re-referencing', time.perf_counter() - t0)


            ######################################## FILTERING ########################################
            t0 = time.perf_counter()

            filt_buffer = self.filt_func(reref_data)


            self.profiler.record('Filtering', time.perf_counter() - t0)


            ################################## THRESHOLD CROSSING ####################################
            t0 = time.perf_counter()

            threshold_crossings = self.get_threshold_crossings(filt_buffer, self.thresholds)

            self.profiler.record('Threshold crossing', time.perf_counter() - t0)


            ################################## SPIKE BAND POWER #####################################
            t0 = time.perf_counter()

            spike_band_power = self.get_spike_bandpower(filt_buffer, self.clip_thresh)


            self.profiler.record('Spike band power', time.perf_counter() - t0)

            ###################################### BIN MULTIPLE #####################################
            t0 = time.perf_counter()

            # do the binning 
            spike_pow_bin = self.bin_data(spike_pow_buffer, bin_type=0)
            thresh_cross_bin = self.bin_data(thresh_cross_buffer, bin_type=1)


            self.profiler.record('Binning', time.perf_counter() - t0)

            ###################################### WRITE TO REDIS #####################################
            t0 = time.perf_counter()


            self.profiler.record('Redis write', time.perf_counter() - t0)



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
                       rev_zi=None,
                       sample_num=None):
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
        n_samp = data.shape[1]
        if not sample_num:
            sample_num = 1
            
        start_idx = -n_samp * sample_num
        end_idx = None if sample_num == 1 else -n_samp * (sample_num - 1)

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
                                                      zi=ic)[0][:, start_idx:end_idx]

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

    nsp_all = NSP_all()
    nsp_all.run()

    gc.collect()  # run garbage collection









