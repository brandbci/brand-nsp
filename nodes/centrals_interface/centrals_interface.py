import gc
import logging
import os
import signal
import sys
import time
from dotenv import load_dotenv

from brand import BRANDNode

SUPERVISOR_COMMANDS_STREAM = 'supervisor_ipstream'
PC1_MATLAB_COMMAND_STREAM = 'pc1_matlab_command'
PC1_MATLAB_RESULT_STREAM = 'pc1_matlab_result'
PC1_MATLAB_DEFAULT_NSP_DIR_NAME = 'NSP_Data'

class CentralsInterface(BRANDNode):
    def __init__(self):
        super().__init__()

        # __file__ doesn't work with cython, so need to specify path
        self.path_to_this_node = os.path.abspath(
            os.path.join(
                '..', 'brand-modules', 'brand-nsp', 'nodes', 'centrals_interface'
            )
        )

        self.parameters.setdefault('metadata_stream', 'block_metadata')

        # Get Window user from environment file created during startup
        self.env_file_path = self.parameters['env_file_path']
        load_dotenv(self.env_file_path)
        pc1_user = os.environ.get('PC1_USER')

        # convert to correct path format if domain name is included for user
        if '\\' in pc1_user:
            pc1_user = pc1_user.split('\\')[1] + '.' + pc1_user.split('\\')[0]

        # Build Windows save path
        self.save_path = os.environ.get('PC1_DATA_DIR')
        if self.save_path is None:
            self.save_path = f'C:\\Users\\{pc1_user}\\projects\\Data'
        else:
            self.save_path = self.save_path.replace('$PC1_USER', pc1_user)

        # read block metadata and wait until the node updates dbfilename
        self.metadata_stream = self.parameters['metadata_stream']
        # get final ID of ms prior to supergraph entry to ensure we xread only this block's metadata
        metadata_id = str(int(self.supergraph_id.split('-')[0])-1)+'-'+str(0xFFFFFFFFFFFFFFFF)
        metadata = self.r.xread({self.metadata_stream: metadata_id}, block=0)
        metadata = metadata[0][1][-1][1]

        self.participant = metadata[b'participant'].decode('utf-8')
        self.session_name = metadata[b'session_name'].decode('utf-8')
        
        rdbfilename = self.r.config_get('dbfilename')['dbfilename']
        self.filepath = rdbfilename.replace('.rdb', '')
        self.filepath = '\\'.join((
            self.save_path,
            self.participant,
            self.session_name,
            PC1_MATLAB_DEFAULT_NSP_DIR_NAME,
            self.filepath))

        self.result_stream_dict = {
            PC1_MATLAB_RESULT_STREAM: '0'
        }
        
        self.redis_timeout = 5000

        self.recording_started = False

    def run(self):

        # get path to MATLAB script we want to run
        matlab_filename = 'start_recording_nsx.m'
        m_path = os.path.join(self.path_to_this_node, matlab_filename)

        # variables to change in the specified MATLAB script
        matlab_parameter_dict = {}
        matlab_parameter_dict['fileout'] = self.filepath

        # Compose matlab script
        matlab_script_str = self.load_matlab_script_as_text(m_path, matlab_parameter_dict)
        logging.info(f'Running: {matlab_script_str}')
        current_timestamp_str = str(time.time())
        self.r.xadd(
            PC1_MATLAB_COMMAND_STREAM,
            {
                'commands': matlab_script_str,
                'command_timestamp': current_timestamp_str # logging the time when the command is sent out
            }
        )

        logging.info(f'Commands specified in <{matlab_filename}> sent to PC1 command stream{" with parameters: " + str(matlab_parameter_dict) if matlab_parameter_dict else ""} (timestamp: {current_timestamp_str}). Waiting for execution.')

        # Check if there is a result from the PC1 MATLAB adapter.
        xread_receive = self.r.xread(self.result_stream_dict, block=self.redis_timeout, count=1)

        if len(xread_receive) == 0:
            logging.warning("No reply received from PC1 MATLAB adapter")

        else:
            entry_id, entry_data = xread_receive[0][1][0]
            logging.info(entry_data[b'result'].decode('utf-8'))
            self.result_stream_dict[PC1_MATLAB_RESULT_STREAM] = entry_id
            self.recording_started = True

        while True:
            signal.pause()
    
    def load_matlab_script_as_text(self, filepath, param_dict=None):
        """
        Load a MATLAB script as text. Note that since the loaded script will be executed by `matlab.engine.eval()`. Thus, the script cannot be a function, all the required parameters will be appended to the script as a string.

        :param str filepath: Path to the MATLAB script file to load.
            e.g. "/path/to/matlabscript.m"
        :return: The text of the MATLAB script.
        :rtype: str
        """

        with open(filepath, "r") as f:
            matlab_script_content = f.read()

        text = matlab_script_content.replace("\n", " ")

        parameter_def_lines = []
        if param_dict is not None:
            for key, value in param_dict.items():
                if isinstance(value, str):
                    value = f"'{value}'"
                elif isinstance(value, (int, float, bool)):
                    value = str(value).lower()
                else:
                    logging.warning(f'Incompatible data type of value received [{str(type(value))}], skipping.')
                    continue
                line = f"{key} = {value};"
                parameter_def_lines.append(line)

            text = " ".join(parameter_def_lines) + " " + text
        return text
    
    def terminate(self, sig, frame):
        if self.recording_started:
            matlab_filename = 'stop_recording_nsx.m'
            filepath = os.path.join(self.path_to_this_node, matlab_filename)

            # Compose matlab script
            matlab_script_str = self.load_matlab_script_as_text(filepath)
            current_timestamp_str = str(time.time())
            self.r.xadd(
                PC1_MATLAB_COMMAND_STREAM,
                {
                    'commands': matlab_script_str,
                    'command_timestamp': current_timestamp_str # logging the time when the command is sent out
                }
            )
            
            # Check if there is a result from the PC1 MATLAB adapter.
            xread_receive = self.r.xread(self.result_stream_dict, block=self.redis_timeout, count=1)

            if len(xread_receive) == 0:
                logging.warning("No reply received from PC1 MATLAB adapter from stop_recording_nsx command")
            
            _, entry_data = xread_receive[0][1][0]
            logging.info(entry_data[b'result'].decode('utf-8'))

        # exit
        BRANDNode.terminate(self, sig, frame)


if __name__ == "__main__":
    gc.disable()
    centrals_interface = CentralsInterface()
    centrals_interface.run()
    gc.enable()
