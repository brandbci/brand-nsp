import gc
import json
import yaml
import logging
import signal
import time
from copy import deepcopy
import scipy
import numpy as np
from brand import BRANDNode
from collections import defaultdict,deque

from helper import get_threshold_crossing, get_spike_bandpower, get_filter_func


class TimingProfiler:
    def __init__(self, enable=False):
        self.timings = defaultdict(list)
        self.enable = enable

    def record(self, operation, duration):
        if self.enable:
            self.timings[operation].append(duration)

    def get_stats(self):
        stats = {}
        for op, times in self.timings.items():
            stats[op] = {
                "mean": np.mean(times),
                "median": np.median(times),
                "min": np.min(times),
                "max": np.max(times),
                "count": len(times),
            }
        return stats

    def print_stats(self):
        if self.enable:
            stats = self.get_stats()
            logging.info("\nTiming Statistics (in milliseconds):")
            logging.info("-" * 85)  # Widened to accommodate median
            logging.info(f"{'Operation':<30} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Count':>10}")
            logging.info("-" * 85)
            for op, metrics in stats.items():
                logging.info(
                    f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['median']*1000:>10.3f} "
                    f"{metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}"
                )
            


class NSP_all(BRANDNode):
    def __init__(self,params=None):
        super().__init__()

        t_init = time.perf_counter()

        if params:
            self.parameters = params # Initialise params from jupyter

        self.init_streams() # Initialise all streams
        self.init_params()  # Initialise all parameters
        

        # Variables for maintaining the latest state (supergraph).
        self.current_supergraph_dict = {}
        self.current_supergraph_id = "0-0"

        self.profiler = TimingProfiler(self.enable_profiler ) # Initialise profiler
        self.profiler.record("INIT", time.perf_counter() - t_init)

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

        # Set flag to indicate that we are  using the correct electrode mapping for this data
        self.r.set("using_correct_electrode_mapping", 1)


    def init_streams(self):

        # misc stream names
        self.input_stream = self.parameters.setdefault("input_stream", "nsp_neural")
        self.coefs_stream_name = self.parameters.setdefault("coefs_stream_name","rereference_parameters")
        self.filt_stream = self.parameters.get('filt_stream', None)
        self.rms_stream = self.parameters.get('adaptive_rms_stream', None)

        # main output stream idx in output_streams
        rrid = self.parameters.get('reref_stream_idx', 0)
        tcid = self.parameters.get('thresh_cross_stream_idx', 1) 
        bpid = self.parameters.get('band_power_stream_idx', 2)
        bsid = self.parameters.get('binned_spikes_stream_idx', 3)

        output_streams = self.parameters.setdefault("output_streams", ["reref_neural", "thresh_cross", "sbp", "binned_spikes"])
        sync_key = self.parameters.setdefault("sync_key", ["tracking_id", "sync", "sync", "sync"])
        ts_key = self.parameters.setdefault("ts_key", ["BRANDS_time", "ts", "ts", "ts"])
        self.td_type = self.parameters.setdefault("td_type", "uint32")

        # Stream assignments
        self.reref_stream = output_streams[rrid]
        self.cross_stream = output_streams[tcid] 
        self.power_stream = output_streams[bpid]
        self.bin_stream = output_streams[bsid]

        # Sync key assignments
        self.reref_sync_key = sync_key[rrid]
        self.cross_sync_key = sync_key[tcid]
        self.power_sync_key = sync_key[bpid]
        self.bin_sync_key = sync_key[bsid]

        # Timestamp key assignments
        self.reref_ts_key = ts_key[rrid]
        self.cross_ts_key = ts_key[tcid]
        self.power_ts_key = ts_key[bpid]
        self.bin_ts_key = ts_key[bsid]

        self.nsp_ts_field = self.parameters.setdefault("nsp_ts_field", "timestamps").encode()
        self.nsp_ts_dtype = self.parameters.setdefault("nsp_ts_dtype", "uint64")
        self.sync_dict_key = self.parameters.setdefault("sync_dict_key", "nsp_idx_1")

    def init_params(self):

        ###### REREF PARAM INIT ######

        self.input_dtype = self.parameters.setdefault("input_dtype", np.int16)
        self.output_dtype = self.parameters.setdefault("output_dtype", 'float32')

        self.samp_freq = self.parameters.setdefault("samp_freq", 30000)
        self.samp_per_stream = self.parameters.setdefault("samp_per_stream", 30)
        self.chan_per_stream = self.parameters.setdefault("chan_per_stream", 256)

        self.start_chan = self.parameters.setdefault("start_channel", 0)
        self.end_chan = self.start_chan + self.chan_per_stream
        self.th_chans = range(self.start_chan, self.end_chan)
        self.n_split = self.parameters.setdefault("n_split", 16)
                
        self.reref_maxlen = self.parameters.setdefault("reref_maxlen", 2000)
        self.neural_data_field = self.parameters.setdefault("neural_data_field", "samples")

        ###### FILTER PARAM INIT ######
        self.initialize_coefficients()

        self.output_filtered = self.parameters.setdefault("output_filtered", False) 
        self.output_reref    = self.parameters.setdefault("output_reref", True) 
        self.acausal_filter = self.parameters.setdefault("acausal_filter", True)  
        self.causal = not self.acausal_filter  # Derived from acausal_filter
        self.acausal_filter_lag = self.parameters.setdefault("acausal_filter_lag", 120)  # Default: 120 samples == 4ms lag

        self.demean = self.parameters.setdefault("enable_CAR", False)

        self.filter_func, self.sos, self.zi, self.rev_win, self.rev_zi = self.build_filter()


        ###### THRESH_CROSS PARAM INIT ######

        self.thresh_mult = self.parameters.setdefault("thresh_mult", -4.5)
        self.pack_per_call = self.parameters.setdefault("pack_per_call", 1)
        self.use_tracking_id = self.parameters.setdefault("use_tracking_id", True)

        self.adaptive_thresholds = self.parameters.setdefault("adaptive_thresholds", False)
        
        if self.adaptive_thresholds:
            self.rms_window_len = self.parameters["adaptive_rms_window_len"]
            self.mean_squared_buffer = np.zeros((self.chan_per_stream, self.rms_window_len), dtype=np.float64)
            self.mean_squared_buffer_index = 0
            self.mean_squared_buffer_full = False
            self.mean_squared_last = np.zeros((self.chan_per_stream), dtype=np.float64)
            self.mean_squared_new = np.zeros((self.chan_per_stream), dtype=np.float64)
            self.root_mean_squared = np.zeros((self.chan_per_stream), dtype=np.float64)
            logging.info(
                f"Adaptive spike thresholds enabled, using RMS computed over {self.samp_per_stream*self.rms_window_len} 30kHz samples"
            )

        self.num_coincident = self.parameters.get("num_coincident_spikes", self.chan_per_stream + 1)

        thresholds_stream = self.parameters.setdefault("thresholds_stream", "thresholds")
        thresholds_file = self.parameters.setdefault("thresholds_file", None)

        self.thresholds = None
        if thresholds_stream is not None:
            self.thresholds = self.load_thresholds_from_stream(thresholds_stream)

        if self.thresholds is None and thresholds_file is not None:
            self.thresholds = self.load_thresholds_from_file(thresholds_file)

        if self.thresholds is None:
            logging.error(f"Thresholds not found in stream={thresholds_stream} or file={thresholds_file}. Exiting...")
            signal.signal(signal.SIGINT, self.terminate)

        self.logscale = self.parameters.setdefault("bandpower_logscale", False)
        self.reduce_sbp = getattr(np, self.parameters.setdefault("sbp_reduction_fn", "mean")) 

        self.bin_size = self.parameters.setdefault("bin_size", 10)
        self.bin_enable = self.parameters.setdefault("bin_enable", True)

        self.enable_profiler = self.parameters.setdefault("enable_profiler", False)


    def load_thresholds_from_file(self, thresholds_file):
        tf_chans = self.th_chans
        try:
            with open(thresholds_file, "r") as f:
                thresh_yaml = yaml.safe_load(f)
            if "thresholds" in thresh_yaml:
                if tf_chans is None:
                    if len(thresh_yaml["thresholds"]) == self.chan_per_stream:
                        logging.info(f"Loaded thresholds from {thresholds_file}")
                        return np.array(thresh_yaml["thresholds"]).reshape(-1, 1)
                    else:
                        raise ValueError(
                            f"Number of thresholds in {thresholds_file} "
                            f'({len(thresh_yaml["thresholds"])}) does not '
                            f"equal chan_per_stream parameter {(self.chan_per_stream)}"
                        )
                # if all of our requested channels are in the available range
                # of channels
                elif (
                    set(tf_chans) & set(range(0, len(thresh_yaml["thresholds"])))
                ) == set(tf_chans):
                    logging.info(f"Loaded thresholds from {thresholds_file}")
                    return np.array(thresh_yaml["thresholds"])[tf_chans].reshape(-1, 1)
                else:
                    raise ValueError(
                        f"Channel range "
                        f'{self.parameters["thresholds_ch_range"]} outside of '
                        f"available channels in {thresholds_file} "
                        f'(max {len(thresh_yaml["thresholds"])})'
                    )
            else:
                logging.warning(f"Could not find 'thresholds' key in {thresholds_file}")
                return None

        except FileNotFoundError:
            logging.warning(f"Could not find thresholds file at {thresholds_file}")
            return None

    def load_thresholds_from_stream(self, stream):
        entry = self.r.xrevrange(stream, "+", "-", count=1)
        th_chans = self.th_chans
        if entry:
            thresholds = np.frombuffer(entry[0][1][b"thresholds"], dtype=np.float64)
            if th_chans is None:
                if len(thresholds) == self.chan_per_stream:
                    logging.info(f"Loaded thresholds from the {stream} stream")
                    return thresholds.reshape(-1, 1)
                else:
                    raise ValueError(
                        f"Number of thresholds in the {stream} stream "
                        f"({len(thresholds)}) does not equal chan_per_stream "
                        f"parameter {(self.chan_per_stream)}"
                    )
            # if all of our requested channels are in the available range of
            # channels
            elif (set(th_chans) & set(range(0, len(thresholds)))) == set(th_chans):
                logging.info(f"Loaded thresholds from the {stream} stream")
                return thresholds[th_chans].reshape(-1, 1)
            else:
                raise ValueError(
                    f'Channel range {self.parameters["thresholds_ch_range"]} '
                    f"outside of available channels in {stream} stream "
                    f"(max {len(thresholds)})"
                )
        else:
            logging.warning(f"{stream} stream has no entries")
            return None

    def initialize_coefficients(self):

        entries = self.r.xrevrange(self.coefs_stream_name, "+", "-", count=1)

        if len(entries):
            logging.info(
                f"Coefficients entry found in Redis stream: {self.coefs_stream_name}"
            )

            entry_id, entry_dict = entries[0]

            self.coefs_all = np.frombuffer(
                entry_dict[b"channel_scaling"], dtype=np.float64
            ).reshape((self.chan_per_stream, self.chan_per_stream))
            self.coefs = self.coefs_all[
                self.start_chan : self.start_chan + self.chan_per_stream,
                self.start_chan : self.start_chan + self.chan_per_stream,
            ]

            if b"channel_unshuffling" in entry_dict:
                unshuffle_all = np.frombuffer(
                    entry_dict[b"channel_unshuffling"], dtype=np.float64
                ).reshape((self.chan_per_stream, self.chan_per_stream))
                logging.info(f"Unshuffling matrix loaded from stream.")
            else:
                unshuffle_all = np.eye(self.chan_per_stream, dtype=np.float64)
                logging.info(
                    f"No unshuffling matrix found. Assuming channels are in order."
                )

            unshuffle = unshuffle_all[
                self.start_chan : self.start_chan + self.chan_per_stream,
                self.start_chan : self.start_chan + self.chan_per_stream,
            ]

            self.coefs = (np.eye(self.chan_per_stream) - self.coefs) @ unshuffle
            self.coefs = self.coefs.astype(self.output_dtype)

        else:
            logging.info(
                f"Coefficients entry not found in Redis stream: {self.coefs_stream_name}. Setting coefficients to compute mean of all channels."
            )

            self.coefs = (
                np.eye(self.chan_per_stream)
                - np.ones((self.chan_per_stream, self.chan_per_stream))
                / self.chan_per_stream
            )
            self.coefs.astype(self.parameters["output_dtype"])

    def build_filter(self):

        but_order = self.parameters.get("butter_order", 4)
        but_low = self.parameters.get("butter_lowercut", 250)
        but_high = self.parameters.get("butter_uppercut", None)

        acausal_filter = self.acausal_filter
        causal = self.causal
        fs = self.samp_freq

        rev_win, rev_zi = None, None

        # determine filter type
        if but_low and but_high:
            filt_type = "bandpass"
            Wn = [but_low, but_high]
        elif but_high:
            filt_type = "lowpass"
            Wn = but_high
        elif but_low:
            filt_type = "highpass"
            Wn = but_low
        else:
            raise ValueError("Must specify 'butter_lowercut' or 'butter_uppercut'")

        # set up filter
        sos = scipy.signal.butter(
            but_order, Wn, btype=filt_type, analog=False, output="sos", fs=fs
        )  # set up a filter
        # initialize the state of the filter
        zi_flat = scipy.signal.sosfilt_zi(sos)
        # so that we have the right number of dimensions
        zi = np.zeros((zi_flat.shape[0], self.chan_per_stream, zi_flat.shape[1]))
        # filter initialization
        for ii in range(self.chan_per_stream):
            zi[:, ii, :] = zi_flat

        # select the filtering function
        if acausal_filter and acausal_filter.lower() == "fir":
            use_fir = True
        else:
            use_fir = False
        filter_func = get_filter_func( causal, use_fir=use_fir)

        # log the filter info
        causal_str = "causal" if causal else "acausal"
        message = f"Loading {but_order :d} order, " f"{Wn} hz {filt_type} {causal_str}"
        if causal:
            message += " IIR filter"
        elif use_fir:
            message += " IIR-FIR filter"
        else:
            message += " IIR-IIR filter"
        message += " with CAR" if self.demean else ""
        logging.info(message)

        if not causal:
            if use_fir:
                # FIR filter (backward)
                N = self.acausal_filter_lag + 1  # length of the filter
                imp = scipy.signal.unit_impulse(N)
                rev_win = scipy.signal.sosfilt(sos, imp)
                # filter initialization
                rev_zi_flat = scipy.signal.lfilter_zi(rev_win, 1.0)
                rev_zi = np.zeros((self.chan_per_stream, rev_zi_flat.shape[0]))
                for ii in range(self.chan_per_stream):
                    rev_zi[ii, :] = rev_zi_flat
            else:
                rev_win = None
                rev_zi = zi.copy()

        return filter_func, sos, zi, rev_win, rev_zi


    def run(self):
        def time_now():
            return np.uint64(time.monotonic_ns()).tobytes()

        ########################################### INIT ###########################################

        rev_win = self.rev_win
        rev_zi = self.rev_zi

        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        sos = self.sos
        thresholds = self.thresholds
        zi = self.zi

        # initialize local arrays
        n_samp = samp_per_stream * pack_per_call
        last_ts = np.array([0])

        buffer_fill = 0  # how many samples have been read into the buffer

        neural_data = np.zeros( (self.chan_per_stream, n_samp), dtype=self.output_dtype)
        neural_data_reref = np.zeros_like(neural_data)

        filt_buffer = np.zeros_like(neural_data)
        rev_buffer = np.zeros((self.chan_per_stream, self.acausal_filter_lag + n_samp), dtype=np.float32)
        samp_times = np.zeros(n_samp, dtype=self.td_type)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=self.td_type)
        
        crossings = np.zeros_like(neural_data_reref)
        cross_now = np.zeros(self.chan_per_stream, dtype=np.int16)
        power_buffer = np.zeros(self.chan_per_stream, dtype=np.float32)

        # init buffers fro binning
        cross_bin_buffer = np.zeros((self.chan_per_stream, self.bin_size), dtype=np.int16)
        power_bin_buffer = np.zeros((self.chan_per_stream, self.bin_size), dtype=np.float32)

        binned_spikes = np.zeros((self.chan_per_stream * 2), dtype=self.output_dtype)

        # initialize stream entries

        reref_dict = {}
        filt_dict = {}
        cross_dict = {}
        rms_dict = {}
        coinc_dict = {}
        power_dict = {}
        bin_dict = {}

        sync_dict = {self.sync_dict_key: int(samp_times[0])}
        # sync_dict_buffer = []
        sync_dict_buffer = deque(maxlen=self.bin_size)

        # set timeout
        timeout = 500
        bin_num=0
        buffer_num = 0
        input_id ='$'
        t_start_bin =0

        while True:

            ########################################### READ FROM REDIS ###########################################

            t0 = time.perf_counter()
            t_start =time.perf_counter()
            xread_receive = self.r.xread(
                {self.input_stream: input_id}, block=timeout, count=1
            )

            if len(xread_receive) >= pack_per_call:
                t_start_after_redis =time.perf_counter()
                input_id = xread_receive[0][1][0][0]
                entry_data = xread_receive[0][1][0][1]

                # read timestamps
                ts = np.concatenate([last_ts, np.frombuffer(entry_data[self.nsp_ts_field], dtype=self.nsp_ts_dtype).astype(int)])

                # check if timestamps are in order
                neg_time_diff = np.diff(ts) < 0
                if np.any(neg_time_diff):
                    neg_ts = ts[1:][neg_time_diff]
                    logging.warning(f"Timestamps {neg_ts} are not in order!!!")
                last_ts = ts[-1:]

                neural_data[:] = (
                    np.frombuffer(entry_data[self.neural_data_field.encode()], dtype=self.input_dtype)
                    .reshape((self.chan_per_stream, self.samp_per_stream))
                    .astype(self.output_dtype)
                )
                if self.use_tracking_id:
                    samp_times[:] = np.repeat(
                        np.frombuffer(entry_data[b"tracking_id"], self.td_type), self.samp_per_stream,)
                else:
                    samp_times[:] = np.frombuffer(entry_data[b"timestamps"], self.td_type)

                self.profiler.record("Redis read", time.perf_counter() - t0)


                ###################################### RE-REFERENCING ######################################


                t0 = time.perf_counter()

                n = 0
                while n < self.chan_per_stream:
                    neural_data_reref[n:n+self.n_split,:] = np.dot(self.coefs[n:n+self.n_split,:], neural_data[:])
                    n += self.n_split

                reref_dict[self.reref_ts_key] = time_now()
                self.profiler.record("Re-referencing", time.perf_counter() - t0)


                ######################################## FILTERING ########################################


                t0 = time.perf_counter()

                if self.causal:
                    self.filter_func(
                        neural_data_reref, filt_buffer, sos, zi
                    )
                else:
                    self.filter_func(
                        neural_data_reref,
                        filt_buffer,
                        rev_buffer,
                        sos=sos,
                        zi=zi,
                        rev_win=rev_win,
                        rev_zi=rev_zi,
                    )

                    samp_times_buffer[:-n_samp] = samp_times_buffer[n_samp:]
                    samp_times_buffer[-n_samp:] = samp_times

                # wait for 4ms/120 (filter lag) before pre-processing for acausal 
                if self.causal:
                    samp_time_current = samp_times[:n_samp]
                else:
                    samp_time_current = samp_times_buffer[:n_samp]
                    if buffer_fill + n_samp < buffer_len: 
                        buffer_fill += n_samp  
                        continue  
                
                filt_dict[self.cross_ts_key] = time_now()
                self.profiler.record("Filtering", time.perf_counter() - t0)


                #################################### THRESHOLD CROSSING ####################################


                t0 = time.perf_counter()

                if self.adaptive_thresholds:
                    # Compute MS for samples corresponding to current ms
                    self.mean_squared_new = np.mean(filt_buffer**2, axis=1)
                    # Update rolling MS iteratively using new and last stored sample
                    self.mean_squared_last += (self.mean_squared_new - self.mean_squared_buffer[:, self.mean_squared_buffer_index]) / self.rms_window_len
                    # Store new MS in buffer, overrtiting oldest sample
                    self.mean_squared_buffer[:, self.mean_squared_buffer_index] = self.mean_squared_new
                    # Circular buffer
                    self.mean_squared_buffer_index += 1
                    if self.mean_squared_buffer_index >= self.rms_window_len:
                        self.mean_squared_buffer_index = 0
                        self.mean_squared_buffer_full = True
                    # Compute RMS
                    self.root_mean_squared = np.sqrt(self.mean_squared_last)
                    # If buffer has filled up, update thresholds using rolling RMS
                    if self.mean_squared_buffer_full:
                        self.thresholds = (self.thresh_mult * self.root_mean_squared).reshape(-1, 1)
                        thresholds = self.thresholds
                    rms_dict[self.cross_ts_key] = time_now()


                cross_now =get_threshold_crossing(crossings,filt_buffer,thresholds,cross_now) #slightly faster
                cross_dict[self.cross_ts_key] = time_now()
                self.profiler.record("Threshold crossing", time.perf_counter() - t0)


                ##################################### SPIKE BAND POWER #####################################
                
                
                t0 = time.perf_counter()

                power_buffer = get_spike_bandpower(filt_buffer, power_buffer, self.logscale) #slightly faster 

                power_dict[self.power_ts_key] = time_now()
                self.profiler.record("Spike band power", time.perf_counter() - t0)


                ##################################### BUFFER AND BINNING #####################################


                cross_bin_buffer[:, buffer_num] = cross_now[:]
                power_bin_buffer[:, buffer_num] = power_buffer[:]

                buffer_num += 1

                if self.bin_enable and self.bin_size == buffer_num:
                    t0 = time.perf_counter()

                    binned_spikes[: self.chan_per_stream] = np.sum(cross_bin_buffer,axis=1)
                    binned_spikes[self.chan_per_stream :] = self.reduce_sbp(power_bin_buffer,axis=1)
                    
                    bin_dict[self.bin_ts_key] = time_now()
                    self.profiler.record("Binning", time.perf_counter() - t0)


                ##################################### WRITE TO REDIS #####################################


                t0 = time.perf_counter()

                p = self.r.pipeline() 

                # write reref_neural if enabled
                if self.output_reref:
                    reref_dict = {k: v for k, v in entry_data.items()}  
                    reref_dict[self.reref_ts_key] = time_now()
                    reref_dict[self.neural_data_field] = neural_data_reref.astype(self.output_dtype).tobytes()
                    p.xadd(self.reref_stream, reref_dict, maxlen=self.reref_maxlen, approximate=True)

                # sync_dict wrt (default:nsp_idx_1)
                sync_dict = {self.sync_dict_key: int(samp_time_current[0])}
                sync_dict_json = json.dumps(sync_dict)
                sync_dict_buffer.append(sync_dict_json)
                
                # check for coincident crossings
                tot_spikes = cross_now.sum()
                if tot_spikes >= self.num_coincident:
                    logging.info(f"{tot_spikes} coincident spikes detected, timestamp: {int(samp_time_current[0])}")

                    coinc_dict[self.cross_sync_key] = sync_dict_json
                    coinc_dict[self.cross_ts_key] = time_now()
                    coinc_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    coinc_dict[b"crossings"] = cross_now.tobytes()
                    p.xadd(f"{self.cross_stream}_coinc", coinc_dict)

                    cross_now[:] = 0

                # write thresh_crossing
                cross_dict[self.cross_sync_key] = sync_dict_json
                cross_dict[b"timestamps"] = samp_time_current[0].tobytes()
                cross_dict[b"crossings"] = cross_now.tobytes()
                p.xadd(self.cross_stream, cross_dict)

                # write filtered data 
                if self.output_filtered:
                    filt_dict[self.cross_sync_key] = sync_dict_json
                    filt_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    filt_dict[b"samples"] = filt_buffer.astype(np.int16).tobytes()
                    p.xadd(self.filt_stream, filt_dict)

                # write adaptive_thresholds
                if self.adaptive_thresholds:
                    rms_dict[self.cross_sync_key] = sync_dict_json
                    rms_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    rms_dict[b"samples"] = self.root_mean_squared.astype(np.float64).tobytes()
                    rms_dict[b"thresholds"] = thresholds.astype(np.float64).tobytes()
                    p.xadd(self.rms_stream, rms_dict)

                # write spike band power sbp 
                power_dict[self.power_sync_key] = sync_dict_json
                power_dict[b"timestamps"] = samp_time_current[0].tobytes()
                power_dict[b"samples"] = power_buffer.tobytes()
                p.xadd(self.power_stream, power_dict)

                # write the binned features
                if self.bin_enable and self.bin_size == buffer_num:

                    bin_dict[self.bin_sync_key] = deepcopy(sync_dict_buffer[0])
                    bin_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    bin_dict[b"samples"] = (binned_spikes.astype(self.output_dtype).tobytes())
                    bin_dict["i"] = np.uint64(bin_num).tobytes()
                    p.xadd(self.bin_stream, bin_dict)

                    buffer_num=0
                    bin_num += 1

                    if t_start_bin: self.profiler.record("Total Binning Time", time.perf_counter() - t_start_bin)
                    t_start_bin = time.perf_counter()

                self.profiler.record("Total Exec Time", time.perf_counter() - t_start_after_redis)
                
                p.execute()
                
                self.profiler.record("Redis write", time.perf_counter() - t0)
                self.profiler.record("Total Time", time.perf_counter() - t_start)
            

            elif len(xread_receive) == 0:
                logging.warning("No neural data has been received in the"
                                f" last {timeout} ms")


    def terminate(self, sig, frame):
        if self.enable_profiler:
            self.profiler.print_stats()

        return BRANDNode.terminate(self, sig, frame)


if __name__ == "__main__":
    gc.disable()  # disable garbage collection

    nsp_all = NSP_all()
    nsp_all.run()

    gc.collect()  # run garbage collection
