# Centrals Interface

Node for interfacing with Blackrock's Centrals software running on the PC1 Windows machine. For example, this node will start and stop Centrals recording `.nsx` neural data files. Currently this is achieved by having `pc1_matlab_adpater` running on PC1 to receive commands sent by `centrals_interface` from Redis and use `cbmex` to control start / stop of the recording.

## Parameters (to specify in graph yaml)

### `"log"` (optional)

- Type: string
- Descriptions:
    - The log level for supervisor to run this node with, e.g. `INFO`.

### `"is_start_recording_requested"` (optional)

- Type: boolean
- Description:
    - When this param is set to `true`, this node starts .nsx neural data recording using an API call to Centrals, and then sets this parameter back to `false`.
    - This param should be used to trigger this behavior during runtime (i.e. whenever), or set in the graph yaml to automatically start recording when the graph is started.
- Default: `false`

### `"is_stop_recording_requested"` (optional)

- Type: boolean
- Description:
    - When this param is set to `true`, this node stops .nsx neural data recording using an API call to Centrals, and then sets this parameter back to `false`.
    - This param should be used to trigger this behavior during runtime (i.e. whenever).
- Default: `false`

### `"recording_save_path"` (required when `is_start_recording_requested` is set to `True`)

- Type: string
- Description:
    - Where `Central` software saves the recorded `.ns5` file on `pc1` machine
    - If this param is not set, the node will skip the execution of `start recording` command

## How it works

1. Change the node parameter `centrals_interface.is_start_recording_requested` (to start recording, parameter `recording_save_path` is required) or `centrals_interface.is_stop_recording_requested` of the node `centrals_interface`.
2. Node `centrals_interface.py` loads the corresponding `MATLAB` file according to the requested action and compose the `MATLAB` command string with the parameters (e.g. `centrals_interface.recording_save_path` to start the recording).
3. Node `centrals_interface` sends the command string to the stream `pc1_matlab_command` and changes the corresponding parameter `centrals_interface.is_<action>_requested` back to `False`. The command is then assigned a timestamp-based ID (for validation of the execution of the command).
4. Script running on PC1 `pc1_matlab_adapter.py` reads the `MATLAB` command string from stream `pc1_matlab_command` and executes it via `matlab.engine.eval()`.
5. Script `pc1_matlab_adapter.py` writes the execution result (successful or unsuccessful) along with the ID of the command to Redis stream `pc1_matlab_result`.
6. Node `centrals_interface` logs the received result of sent command (by checking the ID) asynchronously and waits for a new command.

Note: the system (`centrals_interface` and `pc1_matlab_adapter`) will regard an unsuccessfully executed command as completed and move to the next command. So the user has the responsibility to check the execution results of commands.

## How to test:
* On PC1
    * turn on NSP and start central
    * copy and run script pc1_matlab_adapter.py
* On brand machine
    * run graph `centrals_interface_test.yaml` which will request start recording on `startGraph`.
    * check the outputs on PC1 and brand machine. The central recording window should pop up on PC1.
    * with `redis-cli`, type command `XADD supervisor_ipstream * commands updateParameters centrals_interface "{\"is_stop_recording_requested\": True}"` or simply use `stopGraph` command.
    * The central recording should stop and the window automatically closes on PC1.

# PC1 MATLAB Adapter

Program that receives MATLAB scripts from `Redis` and execute via a `MATLAB` session via `matlab.engine` API

## How to run `pc1_matlab_adapter.py` by a click on PC1 (Windows)
* On the Desktop, right click and select `New > Shortcut`
* In the text box, type in `cmd.exe /k python <path_to_script>/pc1_matlab_adapter.py -n matlab_adapter -i 192.168.150.2`
    * Replace the path to `pc1_matlab_adapter.py` with wherever it is on the system you're using.
    * the nick name `-n` can be changed to any name. It is only used for logging purposes.
    * The IP address `-i` points to the IP address of the Redis server.
* Click Next, on the next window, specify the name of the shortcut.
* Done
