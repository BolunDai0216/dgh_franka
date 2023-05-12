import json
import select
import threading
import time
from time import sleep

import dynamic_graph_manager_cpp_bindings
import lcm
import numpy as np
import yaml
import zmq  # Import the communication libs for connecting to the plotJuggler
import os

from dgh_franka.ipc_trigger_t import ipc_trigger_t

def getDataPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir

class FrankaDynamicGraphHead:
    def __init__(
        self,
        robot_id,
        interface_type,
        plotting=False,
        plotter_port=5555,
    ):
        
        self.plotter_port = plotter_port
        self.plotting = plotting
        self.state = None
        self.trigger_timestamp = 0
        # Get the path and names of the interface yaml files 
        config_files_path = \
        os.path.join(getDataPath(), 'interface_configs')
        config_files = os.listdir(config_files_path)
        # Extract the defined interfaces types
        self.defined_interfaces = \
        [f.split('fr3_')[1].split('.yaml')[0] for f in config_files]
        # Make sure the chosen interface is defined
        assert interface_type in self.defined_interfaces, \
        f'interface can onely be {[i for i in self.defined_interfaces]}'
        robot_interface_config = os.path.join(config_files_path,f'fr3_{interface_type}.yaml')
        #Extract the robot information from the config file        
        with open(robot_interface_config, 'r') as f:
            robot_config = yaml.safe_load(f)

        self.robot_config = robot_config
        robot_name = robot_config['device']['name']
        self.lcm_trigger_toppic = f'{robot_name}_trigger'
        self.cmd_dim = \
        [v['size'] for v in robot_config['device']['controls'].values()][1]
        self.robot_sensors = list(robot_config['device']['sensors'].keys())
        self.robot_cmd = list(robot_config['device']['controls'].keys())[1]

        # Instantiate a Dynamic Graph Head (DGH) class to connect to the robot
        self.head = dynamic_graph_manager_cpp_bindings.DGMHead(robot_interface_config)

        # Threading Interface for getting sync triggers from the DGM over LCM
        self.trigger_msg = ipc_trigger_t()
        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe(
            self.lcm_trigger_toppic, self.trigger_callback
        )
        self.subscription.set_queue_capacity(1)
        self.running = True
        self.lcm_thread = threading.Thread(target=self.LCMThreadFunc)
        self.lcm_thread.start()
        # Enable the ZMQ interface for the PlotJuggler
        if self.plotting:
            # Interface for plotting and logging the data
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.plotter_port}")
        sleep(0.2)
        print("Interface Running...")


    def LCMThreadFunc(self):

        while self.running:
            rfds, wfds, efds = select.select([self.lc.fileno()], [], [], 0.5)
            if rfds: # Handle only if there are data in the interface file
                self.lc.handle()

    def trigger_callback(self, channel, data):
        msg = ipc_trigger_t.decode(data)
        self.trigger_timestamp = \
            np.array(msg.timestamp).reshape(1, 1) / 1000000
        self.update()
    
    def update(self):
        # Get the sensor values from the shared memory
        self.head.read()
        self.state = {sensor:self.head.get_sensor(sensor).copy() 
                 for sensor in self.robot_sensors}
        
    def readJoints(self):
        if time.time()-self.trigger_timestamp > 0.2:
            self.state = None
            return None
        else:
            return self.state

    def setCommand(self, cmd):
        self.head.set_control(self.robot_cmd, cmd.reshape(self.cmd_dim, 1))
        self.head.set_control("ctrl_stamp", np.array(self.trigger_timestamp).reshape(1, 1))
        self.head.write()
        self.cmd_log = cmd
    
    def plotterUpdate(self):
        assert self.plotter, 'Plotter interface is not enabled.'
        if self.state is not None:
            state = {k:v.tolist() for k,v in self.state.item()}
            data = {
                    "timestamp": self.trigger_timestamp,
                    "cmd": self.cmd_log.tolist(),
                    "robot_states": state,
                    }
            self.socket.send_string(json.dumps(data))

    def close(self):
        self.running = False
        self.controller = None
        self.lcm_thread.join()
        self.lc.unsubscribe(self.subscription)
        if self.plotting:
            self.socket.close()
        del self.lc
        del self.head
        print("Interface Closed.")