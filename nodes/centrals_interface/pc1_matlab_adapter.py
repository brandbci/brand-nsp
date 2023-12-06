import os
import sys
import time
import redis
import signal
import logging
import argparse
import matlab.engine

from redis import Redis, ConnectionError, ResponseError 

PC1_MATLAB_COMMAND_STREAM = 'pc1_matlab_command'
PC1_MATLAB_RESULT_STREAM = 'pc1_matlab_result'

class StandaloneBrandNode:
    """
    The life cycle of a standalone BRAND node is not controlled by the supergraph, so all the parameters shoud be defined in the node itself. The only connection with the node and BRAND system is that the node can read and write to Redis.
    """

    def __init__(self):

        # parse input arguments
        argp = argparse.ArgumentParser()
        argp.add_argument('-n', '--nickname', type=str, required=True, default='default_standalone_node')
        argp.add_argument('-i', '--redis_host', type=str, required=True, default='localhost')
        argp.add_argument('-p', '--redis_port', type=int, required=False, default=6379)
        argp.add_argument('-s', '--redis_socket', type=str, required=False)
        argp.add_argument('-l', '--log_level', type=str, required=False, default='INFO')
        args = argp.parse_args()

        len_args = len(vars(args))
        if(len_args < 3):
            print("Arguments passed: {}".format(len_args))
            print("Please check the arguments passed")
            sys.exit(1)

        self.NAME = args.nickname
        redis_host = args.redis_host
        redis_port = args.redis_port
        redis_socket = args.redis_socket
        log_level = args.log_level

        # connect to Redis
        self.r = self.connectToRedis(redis_host, redis_port, redis_socket)

        # set up logging
        numeric_level = getattr(logging, log_level.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log_level)

        logging.basicConfig(format=f'%(asctime)s [{self.NAME}] %(levelname)s: %(message)s',
                            level=numeric_level)

        signal.signal(signal.SIGINT, self.terminate)

    def connectToRedis(self, redis_host, redis_port, redis_socket=None):
        """
        Establish connection to Redis and post initialized status to respective Redis stream
        If we supply a -h flag that starts with a number, then we require a -p for the port
        If we fail to connect, then exit status 1
        # If this function completes successfully then it executes the following Redis command:
        # XADD nickname_state * code 0 status "initialized"        
        """

        #redis_connection_parse = argparse.ArgumentParser()
        #redis_connection_parse.add_argument('-i', '--redis_host', type=str, required=True, default='localhost')
        #redis_connection_parse.add_argument('-p', '--redis_port', type=int, required=True, default=6379)
        #redis_connection_parse.add_argument('-n', '--nickname', type=str, required=True, default='redis_v0.1')

        #args = redis_connection_parse.parse_args()
        #len_args = len(vars(args))
        #print("Redis arguments passed:{}".format(len_args))

        try:
            if redis_socket:
                r = Redis(unix_socket_path=redis_socket)
                print(f"[{self.NAME}] Redis connection established on socket:"
                      f" {redis_socket}")
            else:
                r = Redis(redis_host, redis_port, retry_on_timeout=True)
                print(f"[{self.NAME}] Redis connection established on host:"
                      f" {redis_host}, port: {redis_port}")
        except Exception as e:
            print(f"[{self.NAME}] Error with Redis connection, check again: {e}")
            sys.exit(1)

        return r
    
    def run(self):
        while True:
            self.work()
    
    def work(self):
        pass
    
    def terminate(self, sig, frame):
        # TODO: log the termination state to Redis?
        logging.info('SIGINT received, Exiting')
        self.r.close()
        self.cleanup()
        #self.sock.close()
        sys.exit(0)

    def cleanup(self):
        # Does whatever cleanup is required for when a SIGINT is caught
        # When this function is done, it wriest the following:
        #     XADD nickname_state * code 0 status "done"
        pass

class MatlabAdapter(StandaloneBrandNode):
    def __init__(self):
        super().__init__()

        # Start MATLAB session, TODO: handle errors (what will happen if the MATLAB session is already running / not properly killed?)
        # os.system('matlab -nosplash -nodesktop -r "matlab.engine.shareEngine(\'MATLABEngineforBRAND\')"')

        self.engine = matlab.engine.start_matlab('-nosplash -nodesktop')
        self.engine.addpath(self.engine.genpath('C:\\Program Files\\Blackrock Microsystems\\NeuroPort-Central-Suite'))

        self.input_stream_dict = {
            PC1_MATLAB_COMMAND_STREAM : '$'
        }
        self.redis_connected = True
    
    def work(self):
        try:
            xread_receive = self.r.xread(self.input_stream_dict, block=100, count=1)
        except Exception as e:
            if self.redis_connected:
                logging.warning('Lost connection to remote Redis instance.')
            self.redis_connected = False
            time.sleep(1)
            return
        else :
            if not self.redis_connected:
                logging.info('Redis connection established')
            self.redis_connected = True
        
        if len(xread_receive) == 0:
            return

        entry_id, entry_data = xread_receive[0][1][0]
        matlab_script_str = entry_data[b'commands'].decode('utf-8')

        try:
            self.engine.eval(matlab_script_str, nargout=0)
        except Exception as e:
            result = f'Unable to execute the received MATLAB script, please check the entry ({entry_id}) in the <{PC1_MATLAB_COMMAND_STREAM}> stream. {str(e)}'
            logging.warning(result) 
            result_code = 1
        else:
            result = f"Successfully executed the received MATLAB script ({entry_id}) in the <{PC1_MATLAB_COMMAND_STREAM}> stream."
            logging.info(result)
            result_code = 0

        # log the result to Redis
        self.r.xadd(
            PC1_MATLAB_RESULT_STREAM, 
            {
                "command_id": entry_id,
                "command_timestamp": entry_data[b'command_timestamp'],
                "result_code": result_code,
                "result": result,
                "timestamp": str(time.time()) # logging the time when the result is returned
            }
        )

        self.input_stream_dict[PC1_MATLAB_COMMAND_STREAM] = entry_id
        time.sleep(0.5)


    def cleanup(self):
        # Close MATLAB session
        self.engine.quit()
        super().cleanup()

if __name__ == '__main__':
    matlab_adapter = MatlabAdapter()
    matlab_adapter.run()
