import gc
import json
import logging
import signal
import sys
import time
from pathlib import Path
from glob import glob
from copy import deepcopy
from brand.redis import xread_count
from numba import jit
import numpy as np
import os
from brand import BRANDNode
from collections import defaultdict,deque

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
                "mean": np.mean(times),
                "min": np.min(times),
                "max": np.max(times),
                "count": len(times),
            }
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print("\nTiming Statistics (in milliseconds):")
        print("-" * 80)
        print(f"{'Operation':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Count':>10}")
        print("-" * 80)
        for op, metrics in stats.items():
            print(
                f"{op:<30} {metrics['mean']*1000:>10.3f} {metrics['min']*1000:>10.3f} {metrics['max']*1000:>10.3f} {metrics['count']:>10}"
            )


class NSP_all(BRANDNode):
    def __init__(self,params=None):
        super().__init__()
        self.profiler = TimingProfiler()
        t_init = time.perf_counter()

        if params:
            self.parameters =params
        # Initialise all parameters
        self.init_streams()
        self.init_params()
        

        # Variables for maintaining the latest state (supergraph).
        self.current_supergraph_dict = {}
        self.current_supergraph_id = "0-0"

        self.profiler.record("INIT", time.perf_counter() - t_init)

        # terminate on SIGINT
        signal.signal(signal.SIGINT, self.terminate)

        # Set flag to indicate that we are  using the correct electrode mapping for this data
        self.r.set("using_correct_electrode_mapping", 1)

    def init_streams(self):
        self.input_stream = self.parameters.setdefault("input_stream", "nsp_neural")
        self.coefs_stream_name = self.parameters.setdefault("coefs_stream_name","rereference_parameters")
        self.filt_stream = self.parameters.get('filt_stream', None)
        self.rms_stream = self.parameters.get('adaptive_rms_stream', None)


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



    def init_params(self):

        #### REREF ######
        self.input_stream = self.parameters.setdefault("input_stream", "nsp_neural")
        self.input_dtype = self.parameters.setdefault("input_dtype", np.int16)
        self.output_dtype = self.parameters.setdefault("output_dtype", 'float32')

        self.samp_freq = self.parameters.setdefault("samp_freq", 30000)
        self.samp_per_stream = self.parameters.setdefault("samp_per_stream", 30)

        self.chan_per_stream = self.parameters.setdefault("chan_per_stream", 256)
        self.n_channels = self.parameters.setdefault("chan_per_stream", 256)
        self.n_channels_total = self.parameters.setdefault("chan_per_stream", 256)

        self.start_chan = self.parameters.setdefault("start_channel", 0)
        self.end_chan = self.start_chan + self.n_channels
        self.n_range = np.arange(self.start_chan, self.end_chan)
        self.n_range = self.n_range.astype(int)
        self.th_chans = range(self.start_chan, self.end_chan)
        
        self.n_split = self.parameters.setdefault("n_split", 16)
        self.neural_data_field = self.parameters.setdefault(
            "neural_data_field", "samples"
        )

        self.ts_field = self.parameters.setdefault("ts_field", "BRANDS_time")

        self.nsp_ts_field = self.parameters.setdefault(
            "nsp_ts_field", "timestamps"
        ).encode()
        self.nsp_ts_dtype = self.parameters.setdefault("nsp_ts_dtype", "uint64")
        self.sync_dict_key = self.parameters.setdefault("sync_dict_key", "nsp_idx_1")
        self.sync_dict_keys = self.parameters.setdefault("sync_dict_keys", ["nsp_idx_1"])

        self.initialize_coefficients()

        # Get parameters with default values
        self.output_filtered = self.parameters.setdefault(
            "output_filtered", False
        )  # Default: don't export filtered data
        self.acausal_filter = self.parameters.setdefault(
            "acausal_filter", True
        )  # Default: don't use acausal filtering
        self.causal = not self.acausal_filter  # Derived from acausal_filter
        self.acausal_filter_lag = self.parameters.setdefault(
            "acausal_filter_lag", 120
        )  # Default: 120 samples == 4ms lag

        self.demean = self.parameters.setdefault("enable_CAR", False)
        self.initialize_car_groups()
        self.initialize_ch_mask()
        # keep only masked channels
        for g_idx in range(len(self.car_groups)):
            self.car_groups[g_idx] = list(
                set(self.car_groups[g_idx]).intersection(set(self.ch_mask))
            )

        if self.causal:
            self.filter_func, self.sos, self.zi = self.build_filter()
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win, self.rev_zi) = (
                self.build_filter()
            )

        #### TX ######

        self.thresh_mult = self.parameters.setdefault("thresh_mult", -4.5)
        self.pack_per_call = self.parameters.setdefault("pack_per_call", 1)
        self.use_tracking_id = self.parameters.setdefault("use_tracking_id", True)

        self.adaptive_thresholds = self.parameters.setdefault(
            "adaptive_thresholds", False
        )
        if self.adaptive_thresholds:
            self.rms_window_len = self.parameters["adaptive_rms_window_len"]
            self.mean_squared_buffer = np.zeros(
                (self.n_channels, self.rms_window_len), dtype=np.float64
            )
            self.mean_squared_buffer_index = 0
            self.mean_squared_buffer_full = False
            self.mean_squared_last = np.zeros((self.n_channels), dtype=np.float64)
            self.mean_squared_new = np.zeros((self.n_channels), dtype=np.float64)
            self.root_mean_squared = np.zeros((self.n_channels), dtype=np.float64)
            logging.info(
                f"Adaptive spike thresholds enabled, using RMS computed over {self.samp_per_stream*self.rms_window_len} 30kHz samples"
            )

        self.num_coincident = self.parameters.get(
            "num_coincident_spikes", self.n_channels + 1
        )

        thresholds_stream = self.parameters.setdefault(
            "thresholds_stream", "thresholds"
        )
        thresholds_file = self.parameters.setdefault("thresholds_file", None)

        self.thresholds = None
        if thresholds_stream is not None:
            self.thresholds = self.load_thresholds_from_stream(thresholds_stream)

        if self.thresholds is None and thresholds_file is not None:
            self.thresholds = self.load_thresholds_from_file(thresholds_file)

        if self.thresholds is None:
            # amount of data to use for threshold calculation
            self.thresh_calc_len = self.parameters.setdefault(
                "thresh_calc_len", 2000
            )  # 2s worth of data by default
            self.thresholds = self.calc_thresh(
                self.input_stream,
                self.thresh_mult,
                self.thresh_calc_len,
                self.samp_per_stream,
                self.n_channels,
                self.sos,
                self.zi,
            )

        self.logscale = self.parameters.setdefault("bandpower_logscale", False)
        self.reref_maxlen = self.parameters.setdefault("reref_maxlen", 2000)
        self.bin_size = self.parameters.setdefault("bin_size", 10)
        self.bin_enable = self.parameters.setdefault("bin_enbale", True)

    def calc_thresh(
        self, stream, thresh_mult, thresh_cal_len, samp_per_stream, n_channels, sos, zi
    ):
        reply = xread_count(
            self.r, stream=stream, startid="$", count=thresh_cal_len, block=0
        )

        _, entries = reply[0]  # get the list of entries

        read_arr = np.empty(
            (n_channels, thresh_cal_len * samp_per_stream), dtype=np.float32
        )
        filt_arr = np.empty(
            (n_channels, thresh_cal_len * samp_per_stream), dtype=np.float32
        )
        # read_times = np.empty((thresh_cal_len * samp_per_stream))

        i_start = 0
        for _, entry_data in entries:  # put it all into an array
            i_end = i_start + samp_per_stream
            read_arr[:, i_start:i_end] = np.reshape(
                np.frombuffer(entry_data[b"samples"], dtype=self.input_dtype),
                (self.n_channels_total, samp_per_stream),
            )[self.n_range, :]
            # read_times[i_start:i_end] = np.frombuffer(
            #     entry_data[b'timestamps'], self.td_type)
            i_start = i_end

        if self.causal:
            self.filter_func(read_arr, filt_arr, sos, zi)
        else:
            if self.demean:
                common_average_reference(read_arr, self.car_groups)
            filt_arr[:, :] = scipy.signal.sosfiltfilt(sos, read_arr, axis=1)

        thresholds = (
            thresh_mult * np.sqrt(np.mean(np.square(filt_arr), axis=1))
        ).reshape(-1, 1)

        # log thresholds to database
        thresolds_enc = thresholds.astype(np.int16).tobytes()
        self.r.xadd(f"{self.NAME}_thresholds", {b"thresholds": thresolds_enc})

        logging.info("Calculated and set thresholds")
        return thresholds

    def load_thresholds_from_file(self, thresholds_file):
        tf_chans = self.th_chans
        try:
            with open(thresholds_file, "r") as f:
                thresh_yaml = yaml.safe_load(f)
            if "thresholds" in thresh_yaml:
                if tf_chans is None:
                    if len(thresh_yaml["thresholds"]) == self.n_channels_total:
                        logging.info(f"Loaded thresholds from {thresholds_file}")
                        return np.array(thresh_yaml["thresholds"]).reshape(-1, 1)
                    else:
                        raise ValueError(
                            f"Number of thresholds in {thresholds_file} "
                            f'({len(thresh_yaml["thresholds"])}) does not '
                            f"equal n_channels parameter {(self.n_channels_total)}"
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
                if len(thresholds) == self.n_channels_total:
                    logging.info(f"Loaded thresholds from the {stream} stream")
                    return thresholds.reshape(-1, 1)
                else:
                    raise ValueError(
                        f"Number of thresholds in the {stream} stream "
                        f"({len(thresholds)}) does not equal n_channels "
                        f"parameter {(self.n_channels_total)}"
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
            ).reshape((self.n_channels_total, self.n_channels_total))
            self.coefs = self.coefs_all[
                self.start_chan : self.start_chan + self.chan_per_stream,
                self.start_chan : self.start_chan + self.chan_per_stream,
            ]

            if b"channel_unshuffling" in entry_dict:
                self.unshuffle_all = np.frombuffer(
                    entry_dict[b"channel_unshuffling"], dtype=np.float64
                ).reshape((self.n_channels_total, self.n_channels_total))
                logging.info(f"Unshuffling matrix loaded from stream.")
            else:
                self.unshuffle_all = np.eye(self.n_channels_total, dtype=np.float64)
                logging.info(
                    f"No unshuffling matrix found. Assuming channels are in order."
                )

            self.unshuffle = self.unshuffle_all[
                self.start_chan : self.start_chan + self.chan_per_stream,
                self.start_chan : self.start_chan + self.chan_per_stream,
            ]

            self.coefs = (np.eye(self.chan_per_stream) - self.coefs) @ self.unshuffle
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

    def initialize_car_groups(self):
        if self.demean and "CAR_group_sizes" in self.parameters:
            car_sizes = self.parameters["CAR_group_sizes"]
            if not isinstance(car_sizes, list):
                if isinstance(car_sizes, int):
                    car_sizes = []
                    ch_count = deepcopy(self.n_channels)

                    # get CAR group sizes of the specified size, until we run
                    # out of channels for the stream
                    while ch_count > 0:
                        car_sizes.append(
                            min([self.parameters["CAR_group_sizes"], ch_count])
                        )
                        ch_count -= self.parameters["CAR_group_sizes"]

            self.car_groups = []
            ch_count = 0
            for g in car_sizes:
                if not isinstance(g, int):
                    raise ValueError(
                        "'CAR_group_sizes' must be a list of 'int's or a "
                        "single 'int', but "
                        f'{self.parameters["CAR_group_sizes"]} was given'
                    )
                self.car_groups.append(np.arange(ch_count, ch_count + g).tolist())
                ch_count += g
        else:
            self.car_groups = [np.arange(0, self.n_channels).tolist()]

        if "exclude_channels" in self.parameters:
            exclude_ch = self.parameters["exclude_channels"]
            if not isinstance(exclude_ch, list):
                if isinstance(exclude_ch, int):
                    exclude_ch = [exclude_ch]
            for c in exclude_ch:
                if not isinstance(c, int):
                    raise ValueError(
                        "'exclude_channels' must be a list of 'int's or"
                        " a single 'int', but "
                        f'{self.parameters["exclude_channels"]} was given.'
                        " Exiting"
                    )
            for c in exclude_ch:
                for g in self.car_groups:
                    if c in g:
                        g.remove(c)

    def initialize_ch_mask(self):
        if "ch_mask_stream" in self.parameters:
            ch_mask_entry = self.r.xrevrange(
                self.parameters["ch_mask_stream"], "+", "-", count=1
            )
            if ch_mask_entry:
                self.ch_mask = np.frombuffer(
                    ch_mask_entry[0][1][b"channels"], dtype=np.uint16
                )
                if self.th_chans is None:
                    logging.warning(
                        f"'ch_mask_stream' was provided but 'thresholds_ch_range' was not, so the incorrect channels may be masked"
                    )
                else:
                    self.ch_mask = np.array(
                        list(set(self.ch_mask).intersection(set(self.th_chans)))
                    )
                    self.ch_mask -= self.th_chans[0]
            else:
                logging.warning(
                    f'\'ch_mask_stream\' was set to {self.parameters["ch_mask_stream"]}, but there were no entries. Defaulting to using all channels'
                )
                self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)
        else:
            self.ch_mask = np.arange(self.n_channels, dtype=np.uint16)

    def build_filter(self):

        but_order = self.parameters.get("butter_order", 4)
        but_low = self.parameters.get("butter_lowercut", 250)
        but_high = self.parameters.get("butter_uppercut", None)

        acausal_filter = self.acausal_filter
        causal = self.causal
        fs = self.samp_freq

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
        zi = np.zeros((zi_flat.shape[0], self.n_channels, zi_flat.shape[1]))
        # filter initialization
        for ii in range(self.n_channels):
            zi[:, ii, :] = zi_flat

        # select the filtering function
        if acausal_filter and acausal_filter.lower() == "fir":
            use_fir = True
        else:
            use_fir = False
        filter_func = get_filter_func(self.demean, causal, use_fir=use_fir)

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

        pack_per_call = self.pack_per_call
        samp_per_stream = self.samp_per_stream
        sos = self.sos
        thresholds = self.thresholds
        zi = self.zi

        # initialize arrays
        n_samp = samp_per_stream * pack_per_call
        last_ts = np.array([0])

        buffer_fill = 0  # how many samples have been read into the buffer

        neural_data = np.zeros( (self.n_channels, self.samp_per_stream), dtype=self.output_dtype)
        neural_data_reref = np.zeros_like(neural_data)

        filt_buffer = np.zeros_like(neural_data)
        rev_buffer = np.zeros((self.n_channels, self.acausal_filter_lag + n_samp), dtype=np.float32)
        samp_times = np.zeros(n_samp, dtype=self.td_type)
        buffer_len = rev_buffer.shape[1]
        samp_times_buffer = np.zeros(buffer_len, dtype=self.td_type)
        
        crossings = np.zeros_like(neural_data_reref)
        cross_now = np.zeros(self.n_channels, dtype=np.int16)
        power_buffer = np.zeros(self.n_channels, dtype=np.float32)

        
        cross_bin_buffer = np.zeros((self.n_channels, self.bin_size), dtype=np.int16)
        power_bin_buffer = np.zeros((self.n_channels, self.bin_size), dtype=np.float32)

        window = np.zeros((self.n_channels * 2, self.bin_size), dtype=self.output_dtype)

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
        t_start_bin =time.perf_counter()
        while True:

            ###################################### READ FROM REDIS ######################################
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

                # neural_data_reref = reref_neural_data(
                #     self.coefs, neural_data, neural_data_reref, self.n_split
                # )
                n = 0
                while n < self.chan_per_stream:
                    neural_data_reref[n:n+self.n_split,:] = np.dot(self.coefs[n:n+self.n_split,:], neural_data[:])
                    n += self.n_split
                self.profiler.record("Re-referencing", time.perf_counter() - t0)

                ######################################## FILTERING ########################################
                t0 = time.perf_counter()

                if self.causal:
                    self.filter_func(
                        neural_data_reref, filt_buffer, sos, zi, self.car_groups
                    )
                else:
                    self.filter_func(
                        neural_data_reref,
                        filt_buffer,
                        rev_buffer,
                        sos=sos,
                        zi=zi,
                        group_list=self.car_groups,
                        rev_win=rev_win,
                        rev_zi=rev_zi,
                    )

                    samp_times_buffer[:-n_samp] = samp_times_buffer[n_samp:]
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

                self.profiler.record("Filtering", time.perf_counter() - t0)

                ################################## THRESHOLD CROSSING ####################################
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


                crossings[:, 1:] = ((filt_buffer[:, 1:] < thresholds) &
                                    (filt_buffer[:, :-1] >= thresholds))
                cross_now = np.any(crossings, axis=1).astype(np.int16)
                # cross_now = get_tx0(crossings,filt_buffer,thresholds,cross_now)
                # cross_now = get_threshold_crossing(filt_buffer, thresholds, cross_now)

                self.profiler.record("Threshold crossing", time.perf_counter() - t0)

                cross_now0 =np.zeros_like(cross_now)
                cross_now1 =np.zeros_like(cross_now)
                cross_now2 =np.zeros_like(cross_now)
                cross_now3 =np.zeros_like(cross_now)
                cross_now4 =np.zeros_like(cross_now)
                t0 = time.perf_counter()
                cross_now0 = get_tx0(crossings, filt_buffer, thresholds, cross_now0)
                self.profiler.record("tx0", time.perf_counter() - t0)
                t0 = time.perf_counter()
                cross_now4 = get_tx01(crossings, filt_buffer, thresholds, cross_now4)
                self.profiler.record("tx01", time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                cross_now1 =get_tx1(crossings, filt_buffer, thresholds, cross_now1)
                self.profiler.record("tx1", time.perf_counter() - t0)

                t0 = time.perf_counter()
                cross_now2 = get_tx2(crossings, filt_buffer, thresholds, cross_now2)
                self.profiler.record("tx2", time.perf_counter() - t0)


                t0 = time.perf_counter()
                cross_now3 = get_tx3(crossings, filt_buffer, thresholds, cross_now3)
                self.profiler.record("tx3", time.perf_counter() - t0)

                assert np.all(cross_now0==cross_now1),f"{cross_now0.shape} vs {cross_now1.shape}"
                assert np.all(cross_now0==cross_now2),f"{cross_now0.shape} vs {cross_now2.shape}"
                assert np.all(cross_now0==cross_now3),f"{cross_now0} \n\nvs {cross_now3}"
                t0 = time.perf_counter()
                cross_now = get_threshold_crossing(filt_buffer, thresholds, cross_now)
                self.profiler.record("tx4", time.perf_counter() - t0)

                t0 = time.perf_counter()
                cross_now5 = get_threshold_crossings(filt_buffer, thresholds.squeeze())
                self.profiler.record("tx5", time.perf_counter() - t0)
                # assert np.all(cross_now0==cross_now5),"messeds"
                ################################## SPIKE BAND POWER #####################################
                t0 = time.perf_counter()

                power_buffer = get_spike_bandpower(filt_buffer, power_buffer, self.logscale)

                self.profiler.record("Spike band power", time.perf_counter() - t0)

                ###################################### BUFFER #####################################

                cross_bin_buffer[:, buffer_num] = cross_now[:]
                power_bin_buffer[:, buffer_num] = power_buffer[:]

                buffer_num += 1

                ###################################### BIN MULTIPLE #####################################

                if self.bin_enable and self.bin_size == buffer_num:
                    t0 = time.perf_counter()

                    window[: self.chan_per_stream, :] = cross_bin_buffer
                    window[self.chan_per_stream :, :] = power_bin_buffer
                    
                    self.profiler.record("Binning", time.perf_counter() - t0)

                ###################################### WRITE TO REDIS #####################################
                t0 = time.perf_counter()


                p = self.r.pipeline() 
                time_now = np.uint64(time.monotonic_ns()).tobytes()
                
                
                # redis write reref
                reref_dict = {
                    k: v for k, v in entry_data.items()
                }  # deepcopy(self.entry_data)?

                reref_dict[self.reref_ts_key] = time_now
                reref_dict["samples"] = neural_data_reref.astype(self.output_dtype).tobytes()
                p.xadd(self.reref_stream, reref_dict, maxlen=self.reref_maxlen, approximate=True)

                # redis write crossing

                # if len(sync_dict_buffer)>self.bin_size:
                #     sync_dict_buffer[:self.bin_size]=sync_dict_buffer[1:]
                sync_dict_json = json.dumps(sync_dict)
                sync_dict_buffer.append(sync_dict_json)

                # coincident spike removal
                tot_spikes = cross_now.sum()
                if tot_spikes >= self.num_coincident:
                    logging.info(f"{tot_spikes} coincident spikes detected, timestamp: {int(samp_time_current[0])}")
                    coinc_dict[self.cross_sync_key] = sync_dict_json
                    coinc_dict[self.cross_ts_key] = time_now
                    coinc_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    coinc_dict[b"crossings"] = cross_now.tobytes()
                    p.xadd(f"{self.cross_stream}_coinc", coinc_dict)

                    cross_now[:] = 0

                cross_dict[self.cross_sync_key] = sync_dict_json
                cross_dict[self.cross_ts_key] = time_now
                cross_dict[b"timestamps"] = samp_time_current[0].tobytes()
                cross_dict[b"crossings"] = cross_now.tobytes()
                p.xadd(self.cross_stream, cross_dict)

                # filtered data stream
                if self.output_filtered:
                    filt_dict[self.cross_sync_key] = sync_dict_json
                    filt_dict[self.cross_ts_key] = time_now
                    filt_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    filt_dict[b"samples"] = filt_buffer.astype(np.int16).tobytes()
                    p.xadd(self.filt_stream, filt_dict)

                if self.adaptive_thresholds:
                    # Update rms_dict for writing to Redis
                    rms_dict[self.cross_sync_key] = sync_dict_json
                    rms_dict[self.cross_ts_key] = time_now
                    rms_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    rms_dict[b"samples"] = self.root_mean_squared.astype(np.float64).tobytes()
                    rms_dict[b"thresholds"] = thresholds.astype(np.float64).tobytes()
                    p.xadd(self.rms_stream, rms_dict)

                # redis write power

                power_dict[self.power_sync_key] = sync_dict_json
                power_dict[self.power_ts_key] = time_now
                power_dict[b"timestamps"] = samp_time_current[0].tobytes()
                power_dict[b"samples"] = power_buffer.tobytes()
                p.xadd(self.power_stream, power_dict)

                # redis write bin_multiple if self.bin_size
                if self.bin_enable and self.bin_size == buffer_num:
                    
                    # sync_dict_bin = {}
                    # for sync_key in self.sync_dict_keys:
                    #     sync_dict_bin[sync_key] = deepcopy(sync_dict_buffer[0])
                    # sync_dict_json_bin = json.dumps(sync_dict_bin)
                    # logging.info(f"json: {sync_dict_json}, buffer:{sync_dict_buffer}")
                    bin_dict[self.bin_sync_key] = deepcopy(sync_dict_buffer[0])
                    bin_dict[self.bin_ts_key] = np.uint64(time.monotonic_ns()).tobytes()
                    bin_dict[b"timestamps"] = samp_time_current[0].tobytes()
                    bin_dict[b"samples"] = (window.sum(axis=1).astype(self.output_dtype).tobytes())
                    bin_dict["i"] = np.uint64(bin_num).tobytes()
                    p.xadd(self.bin_stream, bin_dict)
                    self.profiler.record("Total Binning Time", time.perf_counter() - t_start_bin)
                    t_start_bin = time.perf_counter()
                    bin_num += 1
                    buffer_num=0
                self.profiler.record("Total Exec Time", time.perf_counter() - t_start_after_redis)
                p.execute()
                
                self.profiler.record("Redis write", time.perf_counter() - t0)
                self.profiler.record("Total Time", time.perf_counter() - t_start)
            
            elif len(xread_receive) == 0:
                logging.warning("No neural data has been received in the"
                                f" last {timeout} ms")



if __name__ == "__main__":
    gc.disable()  # disable garbage collection

    nsp_all = NSP_all()
    nsp_all.run()

    gc.collect()  # run garbage collection
