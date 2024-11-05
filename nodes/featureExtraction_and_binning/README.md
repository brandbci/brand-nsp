# Feature Extraction AND binning

Receives neural data as input, and calculates features (such as threshold crossings and spikepower) as output, which go into decoders. To calculate these features, this node uses parameters (such as each channel's threshold that defines what a threshold crossing is) which may be supplied or updated by other nodes. Also performs binning of features.

## Output

This node writes entries to the `"binnedFeatures"` redis stream. Entries include threshold crossings (for each channel) and spike band power (for each channel).

## Parameters (to specify in graph yaml)

### `"log"`

- Type: string
- Description: The log level for supervisor to run this node with, e.g. `INFO`. Required because this node inherits from `BRANDNode` which causes it to get a `KeyError` if the `log` parameter is not specified.

### `"input_stream"`

- Type: object
- Description: Holds multiple parameters related to the input stream.

### `"input_stream.name"`

- Type: string
- Description: Name of the redis stream this node reads neural data from.

### `"input_stream.samp_per_stream"`

- Type: int
- Description: How many timesteps (samples) of neural data are represented in each redis entry.

### `"input_stream.chan_per_stream"`

- Type: int
- Description: Total number of channels we are calculating features for (should be the number of arrays, multiplied by the number of channels per array).

### `"input_stream.samp_freq"`

- Type: float
- Description: Value (in Hz) of the frequency the neural data was sampled at. Relevant when filtering the neural signals.

binned_output_stream
continuous_filtered_output_stream

### `binned_output_stream` (optional)

- Type: string
- Description: Name of the redis stream this node writes features to.
- Default: `"binnedFeatures"`

### `continuous_filtered_output_stream` (optional)

- Type: string
- Description: Name of the redis stream this node writes filtered neural data to.
- Default: `"continuousNeural_filtered"`

### `"spike_pow_clip_thresh"`

- Type: float
- Description: Max value we allow the spike band power feature to be.

### `"n_arrays"`

- Type: int
- Description: Number of multielectrode arrays the neural data is coming from. For example, this might be `4`, and `"n_electrodes_per_array"` might be `64`, and the total number of electrodes would be 256.

### `"n_electrodes_per_array"`

- Type: int
- Description: Number of electrodes per microelectrode array used to collect the neural data coming in. For example, this might be `64`, `"n_arrays"` might be `4`, and the total number of electrodes would be 256.

### `"butter_order"`

- Type: int
- Description: Order of the Butterworth bandpass filter applied to each channel's raw neural signal.

### `"butter_lowercut"`

- Type: int
- Description: Lower cutoff frequency of the Butterworth bandpass filter applied to each channel's raw neural signal (e.g. `250`).

### `"butter_uppercut"`

- Type: int
- Description: Upper cutoff frequency of the Butterworth bandpass filter applied to each channel's raw neural signal (e.g. `5000`).

### `"bin_size_ms"`

- Type: int
- Description: Number of 1ms timesteps to bin together.

### `"pack_per_call"` (optional)

- Type: int
- Description: Number of redis entries to fetch from the input stream during each loop of this node. This node waits for this number of entries to be available on the input stream before performing its calculation. It is a tradeoff of immediate latency for long-term throughput. Say `pack_per_call` is `5`. This batching can help with performance if this node is having trouble keeping up with its input (batching 5 calculations together takes less than 5 times as long), but it will be up to 5 ms after a timestep of neural data that that timestep's features are available.
- Default: `1`

### `"threshold_filename"` (optional)

- Type: string
- Description:
    - File path to a JSON file, with a key `"thresholds"` whose value is a list of channel thresholds. This list of thresholds is used to initially populate the `"thresholds"` parameter described below.
    - If not specified, no thresholds are loaded initially.

### `"electrode_map_file"` (optional)

- Type: string
- Description:
    - Filepath to file holding the mapping between channel index (as in the NSP output stream) and electrode index (as in the physical location of the electrode on the Utah array).
    - The file is a JSON file with key `"electrode_mapping"`, whose value is a list (in "electrode order") of numbers representing channel indices, e.g. `[5, 3, 1, 2, 4]`.
    - The channel indices are MATLAB-style, i.e. 1-indexed, not 0-indexed.
    - Putting it all together for the example above, describing everything as 0-indexed, electrode 0's voltages are on channel index 4, electrode 1's voltages are on channel index 2, electrode 2's voltages are on channel index 0, etc.

### `write_filtered_neural` (optional)

- Type: bool
- Description: If `true`, we save the filtered neural signal as a field in this node's output stream (i.e. the neural signal after bandpass and CAR filtering). It is a lot of data (similar to the raw neural data), so by default we do not store this. But when we want to see the exact signal which features are being calculated from (for example, what signal are we checking if it crosses thresholds?) this may be useful.
- Default: `false`

### `"lrr_weights_filename"` (optional)

- Type: string
- Description:
    - File path to a .mat file, with linear regression referencing (LRR) weights that were pre-computed and saved by the `parameterUpdater.py` script. These weights are used to perform LRR.
    - If not specified, no LRR weights will be loaded, and common average referencing (CAR) will be used instead.

### `"channels_to_zero_fname"` (optional)
- NOTE: THIS FEATURE CURRENTLY DOES NOT WORK. COMMENTED OUT IN CODE.
- Type: string
- Description:
    - File path to a JSON file, with a key `"channels_to_zero"` whose value is a list of indices. These index values are MATLAB-style (i.e. `1` is the lowest index). The values of these channels will be set to zero before calculating features.
    - If not specified, no channels will be zeroed out.

### `"thresholds"` (state)

- Type: float[]
- Description: Array of floats, whose length equals the total number of channels. Likely not set in the graph yaml, but rather gets updated while this node is running.
