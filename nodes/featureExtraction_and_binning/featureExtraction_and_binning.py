#!/usr/bin/env python
# -*- coding: utf-8 -*-
# featureExtraction.py
# Maitreyee Wairagkar, Nov 2022 [mwairagkar@ucdavis.edu]

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

class FeatureExtraction_and_binning(BRANDNode):

    def __init__(self, parameters=None):
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

        # parameters of the input stream
        self.input_params = self.parameters['input_stream']

        # continuous neural data stream
        self.input_stream = self.input_params['name']

        # number of samples per channel per redis entry
        self.n_samples = self.input_params['samp_per_stream']

        # number of channels
        self.n_channels = self.input_params['chan_per_stream']

        # sampling rate og raw data 
        self.fs = self.input_params['samp_freq']

        # output redis stream to write features to
        self.binned_output_stream = self.parameters.get('binned_output_stream', 'binnedFeatures')
        self.continuous_filtered_output_stream = self.parameters.get('continuous_filtered_output_stream', 'continuousNeural_filtered')

        # number of Redis packets to read on each iteration
        self.pack_per_call = self.parameters.get('pack_per_call', 1)

        # spike power clip threshold
        self.clip_thresh = self.parameters['spike_pow_clip_thresh']

        # number of arrays
        self.n_arrays = self.parameters['n_arrays']

        # number of electrodes per array (assuming each array has same numer of electrodes)
        self.n_electrodes_per_array = self.parameters['n_electrodes_per_array']

        # number of 1ms windows to use for filtering (this currently works if pack_per_call = 1)
        self.num_win_filtering = self.parameters.get('num_win_filtering', 1)

        # whether or not to write the filtered neural data to redis (useful for parameterUpdater)
        self.write_filtered_neural = self.parameters.get('write_filtered_neural', True)

        # get number of loops after which to update the thresholds
        self.threshold_update_loops = self.parameters.get('threshold_update_loops', -1)

        self.use_numba = self.parameters.get('use_numba', True)

        if self.use_numba:
            import NeuralFeatureExtractor as nfx

        else:
            import NeuralFeatureExtractor_nonumba as nfx
 
        # build filter - non-causal Butterworth filter 
        self.filt_param = {}
        self.filt_param['b'], self.filt_param['a'] = nfx.build_filter(self.parameters['butter_order'], 
                                                                      self.parameters['butter_lowercut'], 
                                                                      self.parameters['butter_uppercut'], self.fs)

        # electrode mapping file name
        self.electrode_map_file = self.parameters.get('electrode_map_file', None)

        # Read electrode mapping if the file name has been provided
        if self.electrode_map_file is not None:
            assert os.path.exists(self.electrode_map_file), 'Electrode mapping file does not exist!'
            self.electrode_mapping = nfx.read_electrode_mapping(self.electrode_map_file)
            logging.info(F'Loaded electrode mapping from file: {self.electrode_map_file}')

        # bin size
        self.bin_size_ms = self.parameters['bin_size_ms']

        # get rereferencing method
        self.rereferencing = self.parameters.get('rereferencing', None)

        # flag to automatically find the latest session with params files
        self.auto_find_params = self.parameters.get('auto_find_params', True)
        self.samba_dir = self.parameters.get('samba_dir', '/samba')
        self.participant = self.parameters.get('participant', 't15')

        # Get the filepath where the LRR weights can be found, if specified directly.
        self.lrr_weights_filename = self.parameters.get('lrr_weights_filename')

        # Get the filepath where the thresholds can be found, if specified directly.
        self.threshold_fname = self.parameters.get('threshold_filename')

        # Use params path and block number to get any params filepaths that were not
        # specified directly.
        self.params_path = self.parameters.get('params_path')
        self.params_block_num = self.parameters.get('params_block_num', -1)

        # variables for configuring warning for when raw neural data = 0 on any channel
        self.zero_data_warning = self.parameters.get('zero_data_warning', False)
        self.zero_data_warning_min_time_ms = self.parameters.get('zero_data_warning_min_time_ms', 10)
        self.zero_data_warning_freq_ms = self.parameters.get('zero_data_warning_freq_ms', 3000)

        if self.zero_data_warning:
            self.zero_data_warning_counter_ms = 0
            self.zero_data_previous_warning_time_ms = 0


        #########################################################################################################
        # ------------------------ Find and load thresholds and optionally LRR weights --------------------------
        #########################################################################################################

        # if params path, lrr weights filename and threshold filename are not provided, find the latest session with params files
        if self.auto_find_params:
            if not self.find_latest_params():
                logging.error(f'Params files could not be automatically found in {os.path.join(self.samba_dir, self.participant)}')
                self.terminate()

        else:
            # no auto-finding-params, so either use the specified params path or specified threshold and lrr weights filenames

            if self.params_path is None and (self.lrr_weights_filename is None or self.threshold_fname is None):
                logging.error('If not automatically searching for params, params path or threshold and lrr_weights filenames must be provided!')
                self.terminate()

            # for thresholds
            if self.threshold_fname is None:
                self.threshold_fname = self.sort_params([str(x) for x in Path(self.params_path).glob('*(*)_thresholds.json')])

                if self.params_block_num != -1:
                    self.threshold_fname = [x for x in self.threshold_fname if f'({self.params_block_num})' in x]

                if len(self.threshold_fname) == 0 and self.params_block_num == -1:
                    logging.error(f'No thresholds file found at: {self.params_path}')
                elif len(self.threshold_fname) == 0:
                    logging.error(f'No thresholds file found for block #{self.params_block_num} at: {self.params_path}')
                else:
                    self.threshold_fname = self.threshold_fname[-1]

            # for lrr weights
            if self.rereferencing == 'lrr' and self.lrr_weights_filename is None:
                self.lrr_weights_filename = self.sort_params([str(x) for x in Path(self.params_path).glob('*(*)_lrr_weights.mat')])

                if self.params_block_num != -1:
                    self.lrr_weights_filename = [x for x in self.lrr_weights_filename if f'({self.params_block_num})' in x]

                if len(self.lrr_weights_filename) == 0 and self.params_block_num == -1:
                    logging.error(f'No LRR weights file found at: {self.params_path}')
                elif len(self.lrr_weights_filename) == 0:
                    logging.error(f'No LRR weights file found for block #{self.params_block_num} at: {self.params_path}')
                else:
                    self.lrr_weights_filename = self.lrr_weights_filename[-1]


        # load thresholds
        self.thresholds = None
        if self.threshold_fname:
            self.thresholds = nfx.read_thresholds_from_json(self.threshold_fname)
            logging.info(F'Loaded thresholds from file: {self.threshold_fname}')
            # Tell the supervisor to update the thresholds in the supergraph.
            self.send_thresholds_update(self.thresholds)
        else:
            logging.error('No thresholds file name provided/found!')

        # load lrr weights if lrr referencing is chosen
        self.lrr_weights = None
        if self.rereferencing == 'lrr':
            if self.lrr_weights_filename:
                self.lrr_weights = nfx.read_lrr_weights(self.lrr_weights_filename)
                logging.info(F'Loaded LRR weights from file: {self.lrr_weights_filename}')
            else:
                logging.error(f'LRR was chosen, but LRR weights for block #{self.params_block_num} was not found at: {self.params_path},'
                              ' and no filepath was specified directly.')

        # Save parameter files as variables in redis, for use in certain derivatives.
        if self.threshold_fname:
            self.r.set('param_thresh', self.threshold_fname)
        if self.lrr_weights_filename:
            self.r.set('param_lrr', self.lrr_weights_filename)
        
        '''
        # channel zeroing file name
        self.channels_to_zero_fname = self.parameters.get('channels_to_zero_fname', None)

        # Read channels to zero if file name has been provided
        if self.channels_to_zero_fname is not None:
            assert os.path.exists(self.channels_to_zero_fname), '[featureExtraction] Channel zeroing file does not exist!'
            self.channels_to_zero = self.get_channels_to_zero()
            print('[featureExtraction] Loaded channels to zero from file: {}'.format(self.channels_to_zero))
        '''

    '''
    def get_channels_to_zero(self):
        # Function to read channels to zero from json file. Returns a list.

        # read channels to zero
        with open(self.channels_to_zero_fname) as f:
            channels_to_zero_dict = json.load(f)

        channels_to_zero = channels_to_zero_dict['channels_to_zero'] - 1 # subtract 1 to make it 0-indexed
    
        return channels_to_zero
    '''


    def find_latest_params(self):

        all_sessions = os.listdir(os.path.join(self.samba_dir, self.participant))
        all_sessions = [session for session in all_sessions if f'{self.participant}.202' in session]
        all_sessions.sort(reverse=True)

        for session in all_sessions:
            session_path = os.path.join(self.samba_dir, self.participant, session)

            if 'params' in os.listdir(session_path):
                params_path = os.path.join(self.samba_dir, self.participant, session, 'params')

                if len(glob(os.path.join(params_path, '*(*)_thresholds.json'))) and len(glob(os.path.join(params_path, '*(*)_lrr_weights.mat'))):
                    # get the latest block number
                    self.params_path = params_path
                    self.threshold_fname = self.sort_params(glob(os.path.join(params_path, '*(*)_thresholds.json')))[-1]
                    self.lrr_weights_filename = self.sort_params(glob(os.path.join(params_path, '*(*)_lrr_weights.mat')))[-1]
                    return True

        return False



    def sort_params(self, params_list):
        # function to sort params list in ascending order of block number

        block = []
        for i in range(len(params_list)):
            block.append(int(params_list[i].split('(')[1].split(')')[0]))

        ind = np.argsort(np.array(block))
        block = np.array(block)[ind]
        params_list = list(np.array(params_list)[ind])

        return params_list


    def load_latest_supergraph(self):
        # Get the latest entry in the supergraph stream in redis and save it for this
        # class to access.

        # Fetch the supergraph from redis.
        SUPERGRAPH_STREAM = 'supergraph_stream'
        supergraph_entries = self.r.xrevrange(SUPERGRAPH_STREAM, count=1)

        # Parse the result from redis.
        supergraph_id, supergraph_entry = supergraph_entries[0]

        # Return quickly if there is no new supergraph.
        if supergraph_id == self.current_supergraph_id:
            return

        # There is a new supergraph, so save it for this class.
        
        supergraph_bytes = supergraph_entry[b'data']
        supergraph_str = supergraph_bytes.decode()
        supergraph_dict = json.loads(supergraph_str)

        self.current_supergraph_id = supergraph_id
        self.current_supergraph_dict = supergraph_dict
    

    def get_latest_thresholds(self):
        # Get the latest supergraph, and get the thresholds parameter from it.

        self.load_latest_supergraph()

        matching_node_dicts = [
            n
            for n in self.current_supergraph_dict['nodes'].values()
            if n['nickname'] == self.NAME
        ]
        if not matching_node_dicts:
            logging.warn(
                f'No parameters entry in supergraph for node \'{self.nickname}\''
            )
        node_dict = matching_node_dicts[0]

        node_params = node_dict['parameters']
        if node_params.get('thresholds') is None:
            return None
        else:
            thresholds = np.array(node_params['thresholds'])
            return thresholds

    def send_thresholds_update(self, thresholds):
        # Send a command to the supervisor to update the thresholds parameter for this
        # node in the supergraph.

        self.r.xadd(
            'supervisor_ipstream',
            {
                'commands': 'updateParameters',
                self.NAME: json.dumps({'thresholds': list(thresholds)}),
                "log_level": "WARNING",
            },
        )

    def run(self):
        t0 = time.perf_counter()
        if self.use_numba:
            import NeuralFeatureExtractor as nfx

        else:
            import NeuralFeatureExtractor_nonumba as nfx
 

        # set timeout
        timeout_ms = 500

        # initialize xread stream dictionary
        input_stream_dict = {self.input_stream: '$'}

        # initialise neural buffer of size num_win_filter x packs_per_call size
        neural_buffer = np.zeros((self.n_samples * self.pack_per_call * self.num_win_filtering, self.n_channels), 
                                dtype=np.int16)  # Here for filtering with historic data, 1 window = 1 ms * packs_per_call

        # initialize tracking id that each output 1ms will receive and iteration counter
        tracking_id = 0
        iteration_counter = 0
        thresh_loop_counter = 0

        # create buffers for feature extracted data
        spike_pow_buffer = np.zeros((self.bin_size_ms, self.n_channels), 
                            dtype=np.float32) 
        thresh_cross_buffer = np.zeros((self.bin_size_ms, self.n_channels), 
                            dtype=np.int16) 
        input_tracking_id_buffer = np.zeros(self.bin_size_ms, dtype=np.uint64)   # store tracking ids of input 1ms samples here 
        filt_30k_buffer = []
        nsp_timestamp_buffer = []
        bin_entries_collected = 0 # bin_entries_collected tracks number of bin entries collected so far

        # Initialise binned output stream
        binned_dat_dict = {}     # stream that saves binned threshold crossings and spikeband power along with timestamps and tracking ids

        self.profiler.record('RUN_INIT', time.perf_counter() - t0)

        while True: 
            binned_dat_dict[b'loop_start_timestamp'] = np.uint64(time.monotonic_ns()).tobytes()

            # Check for new thresholds every 'threshold_update_loops'. Useful when there is a feature updater node
            # if threshold_update_loops is -1, skip checking for new thresholds, useful when you want to only use the pre computed thresholds from file
            if thresh_loop_counter == self.threshold_update_loops:
                # get the latest thresholds, if they've been updated.
                self.thresholds = self.get_latest_thresholds()
                thresh_loop_counter = 0
            elif self.threshold_update_loops != -1:
                thresh_loop_counter += 1

            # if we don't have thresholds yet, do nothing.
            if self.thresholds is None:
                time.sleep(timeout_ms / 1000)
                continue

            # Get data from the input redis stream.

            input_entries = []
            
            t0 = time.perf_counter()

            try_until = time.time() + (timeout_ms / 1000)
            while time.time() < try_until:
                remaining_timeout_ms = int((try_until - time.time()) * 1000)
                remaining_entries_needed = self.pack_per_call - len(input_entries)
                xread_receive = self.r.xread(input_stream_dict,
                                             block=remaining_timeout_ms,
                                             count=remaining_entries_needed)
                
                if len(xread_receive) > 0:           
                    # If we got any entries, add them to our list.
                    input_entries.extend(xread_receive[0][1])

                    # Update our latest entry_id so far.
                    input_stream_dict[self.input_stream] = input_entries[-1][0]

                    # If we have enough entries, move on.
                    if len(input_entries) == self.pack_per_call:
                        break

            self.profiler.record('Redis read', time.perf_counter() - t0)
            # only run this if we got input data
            if len(input_entries) == self.pack_per_call:  
                # --------------- I. Concatenate input entries ------------------------------------

                # Concatenate the lists of samples from each redis entry along axis 0 (time)
                # so they can be preprocessed in one go.
                neural_data = np.zeros((self.n_samples * self.pack_per_call, self.n_channels), dtype='int16')
                sample_times = np.zeros((self.n_samples * self.pack_per_call), dtype='uint64')

                for i, (entry_id, entry_data) in enumerate(input_entries):
                    neural_data[(i*self.n_samples):(i*self.n_samples)+self.n_samples,:] = np.reshape(
                                                                                          np.frombuffer(entry_data[b'samples'], np.int16),
                                                                                          (self.n_samples, self.n_channels)
                                                                                          )
                    sample_times[(i*self.n_samples):(i*self.n_samples)+self.n_samples] = np.frombuffer(entry_data[b'timestamps'], np.uint64)
                # ------------------------------------------------------------------------
                    

                # --------------- Ia. Check for zero-valued data, warn if so ----------------------------
                if self.zero_data_warning:
                    
                    # Check if we've gone long enough without a warning to send another one.
                    current_time_ms = time.time() * 1000
                    if current_time_ms - self.zero_data_previous_warning_time_ms >= self.zero_data_warning_freq_ms:

                        # Sum across time axis and check if any channel has all zeros for {self.zero_data_warning_min_time_ms} consecutive ms.
                        if np.any(np.all(neural_data == 0, axis=0)):
                            self.zero_data_warning_counter_ms += self.pack_per_call
                        else:
                            self.zero_data_warning_counter_ms = 0

                        # If we've seen all zeros for long enough, send a warning.
                        if self.zero_data_warning_counter_ms >= self.zero_data_warning_min_time_ms:
                            logging.error(f'Raw neural data is zero-valued on at least one channel for at least {self.zero_data_warning_min_time_ms} ms!')
                            self.zero_data_previous_warning_time_ms = current_time_ms
                            self.zero_data_warning_counter_ms = 0
                # ------------------------------------------------------------------------


                # --------------- II. Perform preprocessing ----------------------------
                
                # Preprocessing is done on all the entries together to save time.
                # (This is where the benefit of pack_per_call comes from.)

                # 0. Reorder channels according to participant specific electrode mapping
                if self.electrode_map_file is not None:
                    neural_data  = nfx.order_channels_using_mapping(neural_data, self.electrode_mapping)
                
                '''
                # 0. Zero out specified channels
                if self.channels_to_zero_fname is not None:
                    neural_data[:, self.channels_to_zero] = 0
                '''

                # 1. Filter the data 
                
                # Insert neural data in a neural buffer so we can use previous windows for more accurate filtering of this window
                neural_buffer[0:-neural_data.shape[0],:] = neural_buffer[neural_data.shape[0]:,:] # shift previous windows to left 
                neural_buffer[-neural_data.shape[0]:,:] = neural_data                             # insert new data on right
                
                # Filter the buffered data
                filt_data = nfx.filter_signal(neural_buffer.copy(), self.n_samples, self.filt_param['b'], self.filt_param['a'], neural_data.shape[0])

                # 2. Rereferencing for noise removal
                if self.rereferencing == 'car':
                    # Common average referencing
                    reref_data = nfx.car(filt_data.copy(),self.n_arrays, self.n_electrodes_per_array)
                elif self.rereferencing == 'lrr':
                    # Linear regression referencing
                    if self.lrr_weights_filename is not None:
                        reref_data = nfx.lrr(filt_data.copy(), self.n_arrays, self.n_electrodes_per_array, self.lrr_weights)
                    else:
                        logging.error('LRR filename not provided! Not doing LRR.')
                else:
                    # No rereferencing
                    reref_data = filt_data.copy()
                # ------------------------------------------------------------------------
                
                # Extracting features is done for each individual input entry, so we
                # don't sacrifice granularity of feature calculation when using
                # pack_per_call for performance reasons.
                for slice_idx in range(self.pack_per_call):
                    slice_start = self.n_samples * slice_idx
                    slice_end   = self.n_samples * (slice_idx + 1)

                    filt_slice = filt_data[slice_start:slice_end] # This is filtered data without rereferencing to be sent to stream
                    data_slice = reref_data[slice_start:slice_end]
                    sample_times_slice = sample_times[slice_start:slice_end]

                    # --------------- III. Perform feature extraction ----------------------------

                    # 3. Extract threshold crossings
                    threshold_crossings = nfx.get_threshold_crossings(data_slice, self.thresholds)

                    # 4. Get spike-band power
                    spike_band_power = nfx.get_spike_bandpower(data_slice, self.clip_thresh)

                    # ------------------------------------------------------------------------


                    # --------------- IV. Write entries to binning buffers ---------------------------------

                    spike_pow_buffer[bin_entries_collected, :] = spike_band_power
                    thresh_cross_buffer[bin_entries_collected, :] = threshold_crossings
                    input_tracking_id_buffer[bin_entries_collected] = tracking_id
                    filt_30k_buffer.append(filt_slice)
                    nsp_timestamp_buffer.append(sample_times_slice)

                    tracking_id += 1
                    bin_entries_collected += 1
                    # ------------------------------------------------------------------------


                    # --------------- IV. Write output stream when buffer is full ------------------------------------
                    # If buffer is full, reset it 
                    if bin_entries_collected == self.bin_size_ms:

                        # do the binning 
                        spike_pow_bin = nfx.bin_data(spike_pow_buffer, bin_type=0)
                        thresh_cross_bin = nfx.bin_data(thresh_cross_buffer, bin_type=1)

                        # reset the buffer and counter
                        bin_entries_collected = 0
                        spike_pow_buffer = np.zeros((self.bin_size_ms, self.n_channels), 
                                dtype=np.float32) 
                        thresh_cross_buffer = np.zeros((self.bin_size_ms, self.n_channels), 
                                dtype=np.int16) 


                        # --------------- Write output stream ------------------------------------
                        # pack into a byte object and put into the corresponding dicts
                        binned_dat_dict[b'spike_band_power'] = spike_pow_bin.astype(np.float32).tobytes()
                        binned_dat_dict[b'threshold_crossings'] = thresh_cross_bin.astype(np.int16).tobytes()

                        # Add synchronisation NSP (cerebus) ids (30 ids) for each 1ms data window used in this bin
                        binned_dat_dict[b'input_nsp_timestamp'] = np.array(nsp_timestamp_buffer).astype(np.uint64).tobytes()

                        # write input and output tracking ids and reset them 
                        binned_dat_dict[b'input_tracking_ID'] = input_tracking_id_buffer.astype(np.uint64).tobytes()
                        binned_dat_dict[b'output_tracking_ID'] = np.uint64(input_tracking_id_buffer[-1]).tobytes() # last index of input is tracking id
                        input_tracking_id_buffer = np.zeros(self.bin_size_ms, dtype=np.uint64) # reset input id buffer

                        binned_dat_dict[b'input_stream_name'] = self.input_stream

                        binned_dat_dict[b'iteration_counter'] = np.uint64(iteration_counter).tobytes() # log and increment the iteration counter
                        iteration_counter += 1

                        # Create a new redis pipeline
                        redis_pipeline = self.r.pipeline()

                        # optionally output filtered 30k data
                        if self.write_filtered_neural:
                            # Add the filtered neural signal, for parameterUpdater to avoid re-filtering raw data
                            # and for debugging (so we can see exactly what we're comparing to thresholds.)
                            redis_pipeline.xadd(self.continuous_filtered_output_stream, {'samples': np.array(filt_30k_buffer).astype(np.int16).tobytes(),
                                                                                         'timestamps': np.array(nsp_timestamp_buffer).astype(np.uint64).tobytes()},
                                                                                         maxlen=int(5*60*1000/self.bin_size_ms))
                            filt_30k_buffer = [] # reset filtered data buffer
                        nsp_timestamp_buffer = [] # reset after optionally writing filtered data to stream

                        # log loop end timestamp (right before redis writing)
                        binned_dat_dict[b'loop_end_timestamp'] = np.uint64(time.monotonic_ns()).tobytes()

                        # Write to the redis streams.
                        redis_pipeline.xadd(self.binned_output_stream, binned_dat_dict) 

                        # Actually send the write to Redis
                        redis_pipeline.execute()
                    # ------------------------------------------------------------------------

            else:
                logging.warning(F'Not getting neural data. In last {timeout_ms} ms, '
                                F'got {len(input_entries)} entries.')

    def terminate(self, sig=None, frame=None):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)

if __name__ == '__main__':
    gc.disable()  # disable garbage collection

    node = FeatureExtraction_and_binning()
    node.run()

    gc.collect()  # run garbage collection
