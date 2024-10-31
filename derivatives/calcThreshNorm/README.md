# How to use the `calcThreshNorm` derivative

## Purpose of the `calcThreshNorm` derivative

This derivative reads raw neural data streams from an open Redis database and computes voltage thresholds, spike rate means, and spike rate standard deviations. It then writes the voltage thresholds, mean spike rates, spike rate standard deviations, and the parameters used to compute them to a file.

## Usage

The `calcThreshNorm` derivative requires the following first four ordered inputs (sent automatically by Emory's `supervisor` implementation):

```plaintext
0: The name of the Redis database that will form the stored filename
1: The Redis host IP address
2: The Redis host port
3: The path to which to save the file (calcThreshNorm automatically generates and saves to the 'thresh_norm' folder at this path)
```

### Parameters:

The following parameters must be set in the supergraph:

```yaml
derivatives:
  - calcThreshNorm:
      name:                 calcThreshNorm.py           # required if autorun below is set to 'True'
      module:               ../brand-modules/brand-nsp  # required if autorun below is set to 'True'
      autorun:              True                        # whether supervisor should automatically run this derivative after 'stopGraph' command
      parameters:
        input_stream_name:  # a single string or a list of stream names to pull data from
        input_stream_key:   # a single string or a list of keys corresponding to the streams above which contain the data
```

The following parameters can be optionally set in the supergraph:

```yaml
        thresh_mult:        # the RMS multiplier to compute voltage thresholds, default -4.5
        filter_first:       # whether to filter the data prior to computing thresholds, default True
        causal:             # whether to filter data with a causal or acausal filter, default False
        butter_order:       # order of the Butterworth filter used to filter the data, default 4
        butter_lowercut:    # lower cutoff frequency of the Butterworth filter in Hz, default 250
        butter_uppercut:    # upper cutoff frequency of the Butterworth filter in Hz, default 5000
        samp_freq:          # sampling frequency of the incoming data in Hz, default 30000
        rereference:        # what type of rereferencing to apply after filtering data, options are CAR, LRR, or None (default)
        reref_group_sizes:  # the size of each rereference group as a value or a list following concatenation of the data ordered by 'input_stream_name', defaults to splitting by the number of channels in each stream
        exclude_channels:   # a single value or a list of channels to exclude from common-average referencing following concatenation of the data ordered by 'input_stream_name', default None
        bin_size:           # the amount of time in ms over which to compute spiking rate normalization parameters
```