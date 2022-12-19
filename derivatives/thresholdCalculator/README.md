# How to use the `thresholdCalculator` derivative

## Purpose of the `thresholdCalculator` derivative

This derivative reads raw neural data streams from an open Redis database and computes voltage thresholds. It then writes the voltage thresholds and the parameters used to compute them to a file.

## Usage

The `thresholdCalculator` derivative requires the following first four ordered inputs (sent automatically by Emory's `supervisor` implementation):

```plaintext
0: The name of the Redis database that will form the stored filename
1: The Redis host IP address
2: The Redis host port
3: The path to which to save the file (thresholdCalculator automatically generates and saves to the 'thresholds' folder at this path)
```

### Parameters:

The following parameters must be set in the supergraph:

```yaml
derivatives:
  - thresholdCalculator:
      name:                 thresholdCalculator.py      # required if autorun below is set to 'True'
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
        enable_CAR:         # whether to common-average reference the data prior to filtering, default True
        CAR_group_sizes:    # the size of each CAR group as a value or a list following concatenation of the data ordered by 'input_stream_name', defaults to splitting by the number of channels in each stream
        exclude_channels:   # a single value or a list of channels to exclude from common-average referencing following concatenation of the data ordered by 'input_stream_name', default None
```