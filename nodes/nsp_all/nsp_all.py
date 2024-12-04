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
        self.bin_size = self.parameters['bin_size']
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




        ##### From Threshold extraction

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

        ########################################### INIT ############################################


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

        samp_times = np.zeros(n_samp, dtype=self.tdtype)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=self.tdtype)
        buffer_fill = 0  # how many samples have been read into the buffer


        neural_data = np.zeros((self.chan_per_stream, self.samp_per_stream), dtype=self.output_dtype)
        neural_data_reref = np.zeros_like(neural_data)

        filt_buffer =  np.zeros_like(neural_data)
        rev_buffer = np.zeros(
            (self.chan_per_stream, self.acausal_filter_lag + n_samp),
            dtype=np.float32)

        cross_now =     np.zeros(self.chan_per_stream,dtype=np.int16)
        power_buffer = np.zeros(self.chan_per_stream, dtype=np.float32)

        buffer_num = 0
        cross_bin_buffer = np.zeros((self.chan_per_stream,self.bin_size), dtype=np.int16)   
        power_bin_buffer = np.zeros((self.chan_per_stream,self.bin_size), dtype=np.float32)   

        window = np.zeros((self.chan_per_stream*2,self.bin_size), dtype=self.output_dtype)   


        # initialize stream entries
        cross_dict = {}
        filt_dict = {}
        rms_dict = {}
        coinc_dict = {}
        power_dict = {}
        sync_dict = {self.sync_source_id: int(samp_times[0])}

    
        # initialize xread stream dictionary
        input_stream_dict = {input_stream: '$'}

        # set timeout
        timeout = 500

        # name the filtered output stream
        filt_stream = f'{self.NAME}_filt'








        while True:

            ###################################### READ FROM REDIS ######################################
            t0 = time.perf_counter()
            
            self.replies = self.r.xread({self.input_stream_name: self.input_id}, 
                                        block=0,
                                        count=1)

            self.input_id = self.replies[0][1][0][0]
            self.entry_data = self.replies[0][1][0][1]
            
            # read timestamps
            ts = np.concatenate([self.last_ts, np.frombuffer(self.entry_data[self.nsp_ts_field], dtype=self.nsp_ts_dtype).astype(int)])
            # check if timestamps are in order
            neg_time_diff = np.diff(ts) < 0
            if np.any(neg_time_diff):
                neg_ts = ts[1:][neg_time_diff]
                logging.warning(f"Timestamps {neg_ts} are not in order!!!")
            self.last_ts = ts[-1:]
            
            neural_data[:] = np.frombuffer(self.entry_data[self.neural_data_field.encode()], 
                                                dtype=self.input_dtype).reshape((self.chan_per_stream, self.samp_per_stream)).astype(self.output_dtype)



            self.profiler.record('Redis read', time.perf_counter() - t0)


            ###################################### RE-REFERENCING ######################################
            t0 = time.perf_counter()

            data_buffer = reref_neural_data(self.coefs, neural_data, neural_data_reref, self.n_split)


            self.profiler.record('Re-referencing', time.perf_counter() - t0)


            ######################################## FILTERING ########################################
            t0 = time.perf_counter()

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

                samp_times_buffer[:-n_samp] = (samp_times_buffer[n_samp:])
                samp_times_buffer[-n_samp:] = samp_times

            # find for each channel along the first dimension, keep dims,
            # pack into a byte object and put into the thresh crossings
            # dict
            if self.causal:
                samp_time_current = samp_times[:n_samp]
            else:
                samp_time_current = samp_times_buffer[:n_samp]
                # check the buffer
                if buffer_fill + n_samp < buffer_len:  # buffer is not full
                    buffer_fill += n_samp  # count the samples added
                    continue  # skip writing to Redis



            self.profiler.record('Filtering', time.perf_counter() - t0)


            ################################## THRESHOLD CROSSING ####################################
            t0 = time.perf_counter()

            if self.adaptive_thresholds:
                # Compute MS for samples corresponding to current ms
                self.mean_squared_new = np.mean(filt_buffer**2, axis=1)
                # Update rolling MS iteratively using new and last stored sample
                self.mean_squared_last += (self.mean_squared_new - self.mean_squared_buffer[:,self.mean_squared_buffer_index])/self.rms_window_len
                # Store new MS in buffer, overrtiting oldest sample
                self.mean_squared_buffer[:,self.mean_squared_buffer_index] = self.mean_squared_new
                # Circular buffer
                self.mean_squared_buffer_index += 1
                if self.mean_squared_buffer_index >= self.rms_window_len:
                    self.mean_squared_buffer_index = 0
                    self.mean_squared_buffer_full = True
                # Compute RMS
                self.root_mean_squared = np.sqrt(self.mean_squared_last)
                # If buffer has filled up, update thresholds using rolling RMS
                if self.mean_squared_buffer_full:
                    self.thresholds = (self.thresh_mult * self.root_mean_squared).reshape(-1,1)
                    thresholds = self.thresholds


            cross_now = get_threshold_crossing(filt_buffer, thresholds, cross_now)

            self.profiler.record('Threshold crossing', time.perf_counter() - t0)


            ################################## SPIKE BAND POWER #####################################
            t0 = time.perf_counter()

            power_buffer = get_spike_bandpower(filt_buffer, power_buffer, self.logscale)


            self.profiler.record('Spike band power', time.perf_counter() - t0)



            ###BUFFER ###############

            cross_bin_buffer[:, buffer_num] = cross_now[:]     
            power_bin_buffer[:, buffer_num] = power_buffer[:]
        
            buffer_num +=1

            ###################################### BIN MULTIPLE #####################################
            t0 = time.perf_counter()

            if self.bin_size == buffer_num:
                
                window[:self.chan_per_stream,:] = cross_bin_buffer.sum(axis=1).astype(self.output_dtype)
                window[self.chan_per_stream:,:] = power_bin_buffer.sum(axis=1).astype(self.output_dtype)
                

            self.profiler.record('Binning', time.perf_counter() - t0)

            ###################################### WRITE TO REDIS #####################################
            t0 = time.perf_counter()
            
            if self.old_nsp_streams:

                p = self.r.pipeline()  # create a new pipeline

                # log timestamps
                time_now = np.uint64(time.monotonic_ns()).tobytes()

                #redis write reref
                for key in self.entry_data.keys():
                    if key.decode() != self.neural_data_field and key.decode() != self.ts_field:
                        self.output_dict[key] = self.entry_data[key]
                    elif key.decode() == self.ts_field:
                        self.output_dict[self.ts_field] = np.uint64(time.monotonic_ns()).tobytes()
                    elif key.decode() == self.neural_data_field:
                        self.output_dict[self.neural_data_field] = self.neural_data_reref.astype(self.output_dtype).tobytes()
            
                p.xadd(self.output_stream_name, self.output_dict)

                #redis write crossing
                
                # coincident spike removal
                tot_spikes = cross_now.sum()
                if tot_spikes >= self.num_coincident:
                    logging.info(f'{tot_spikes} coincident spikes detected, timestamp: {int(samp_time_current[0])}')
                    coinc_dict[self.sync_key] = json.dumps(sync_dict)
                    coinc_dict[b'timestamps'] = samp_time_current[0].tobytes()
                    coinc_dict[self.time_key] = time_now
                    coinc_dict[b'crossings'] = cross_now.tobytes()
                    p.xadd(f'{self.NAME}_coinc', coinc_dict)
                    cross_now[:] = 0
                    
                cross_dict[b'crossings'] = cross_now.tobytes()

                # log timestamps
                cross_dict[self.time_key] = time_now

                # thresholdCrossings stream
                p.xadd(self.NAME, cross_dict)
                # filtered data stream
                if output_filtered:
                    # if we're storing the filtered data
                    filt_dict[self.sync_key] = json.dumps(sync_dict)
                    filt_dict[b'timestamps'] = samp_time_current.tobytes()
                    filt_dict[b'samples'] = filt_buffer.astype(
                        np.int16).tobytes()
                    filt_dict[self.time_key] = time_now
                    # add the filtered stuff to the pipeline
                    p.xadd(filt_stream, filt_dict)
                if self.adaptive_thresholds:
                    # Update rms_dict for writing to Redis
                    rms_dict[self.sync_key] = json.dumps(sync_dict)  
                    rms_dict[b'timestamps'] = samp_time_current[0].tobytes()    
                    rms_dict[b'samples'] = self.root_mean_squared.astype(
                        np.float64).tobytes()  
                    rms_dict[b'thresholds'] = thresholds.astype(np.float64).tobytes() 
                    rms_dict[self.time_key] = time_now   
                    p.xadd(self.adaptive_rms_stream, rms_dict)             



                #redis write power
                
                
                # filtered data stream
                # if we're storing the filtered data
                power_dict[self.sync_key] = json.dumps(sync_dict)
                power_dict[b'timestamps'] = samp_time_current[0].tobytes()
                power_dict[b'samples'] = power_buffer.tobytes()
                power_dict[self.time_key] = time_now
                # add the filtered stuff to the pipeline
                p.xadd(power_stream, power_dict)


                
                if self.bin_size == buffer_num:
                #redis write bin_multiple if self.bin_size
                    sync_dict = {}
                    for stream in self.sync_entries:
                        sync_entry_dict = stream[0]  # first entry from each stream
                        for key in sync_entry_dict:
                            sync_dict[key] = sync_entry_dict[key]
                    sync_dict_json = json.dumps(sync_dict)

                    # write results to Redis
                    self.output_entry[self.time_key] = np.uint64(
                        time.monotonic_ns()).tobytes()
                    self.output_entry[self.sync_key] = sync_dict_json
                    self.output_entry['samples'] = self.window.sum(axis=1).astype(
                        self.out_dtype).tobytes()
                    self.output_entry['i'] = np.uint64(self.i).tobytes()

                    self.r.xadd(self.output_stream, self.output_entry)

                    self.i += 1
            
            
            elif self.bin_size == buffer_num:
                #redis nested data with common timestamps
                sync_dict = {}
                for stream in self.sync_entries:
                    sync_entry_dict = stream[0]  # first entry from each stream
                    for key in sync_entry_dict:
                        sync_dict[key] = sync_entry_dict[key]
                sync_dict_json = json.dumps(sync_dict)

                # write results to Redis
                self.output_entry[self.time_key] = np.uint64(
                    time.monotonic_ns()).tobytes()
                self.output_entry[self.sync_key] = sync_dict_json
                self.output_entry['samples'] = self.window.sum(axis=1).astype(
                    self.out_dtype).tobytes()
                self.output_entry['i'] = np.uint64(self.i).tobytes()

                self.r.xadd(self.output_stream, self.output_entry)

                self.i += 1




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









