import json
import threading
from time import sleep

import dynamic_graph_manager_cpp_bindings
import lcm
import numpy as np
import zmq  # Import the communication libs for connecting to the plotJuggler

from dgh_franka.ipc_trigger_t import ipc_trigger_t


class FrankaDynamicGraphHead:
    def __init__(
        self,
        robot_config,
        ds_ratio=10,
        plotting=True,
        logging=True,
        max_log_count=1e7,
        plotter_port=5555,
    ):
        self.controller = None
        self.plotter_port = plotter_port
        self.robot_config = robot_config
        self.ds_ratio = ds_ratio
        self.trigger_counter = 0
        self.logging = logging
        self.max_log_count = max_log_count
        self.logging = logging
        self.plotting = plotting
        self.log_data = []
        self.pause = True
        self.cmd_log = np.zeros(7)

    def thread(self):
        while self.running:
            self.lc.handle()

    def start(self):
        print("Starting the Thread ...")
        # initial_states can be used by the controller
        # Instantiate a Dynamic Graph Head (DGH) class to connect to the robot
        self.head = dynamic_graph_manager_cpp_bindings.DGMHead(self.robot_config)
        self.log_data = []
        self.trigger_counter = 0
        self.initial_states = self.read_states()

        if self.plotting:
            # Interface for plotting and logging the data
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.plotter_port}")

        # Interface for getting sync triggers from the DGM over LCM
        self.msg = ipc_trigger_t()
        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe(
            "dgm_franka_control_trigger", self.trigger_callback
        )
        self.subscription.set_queue_capacity(1)

        self.running = True
        self.lcm_thread = threading.Thread(target=self.thread)
        self.lcm_thread.start()
        sleep(0.2)
        print("Tread Started!")

    def close(self):
        print("Stopping the thread ...")
        self.running = False
        self.controller = None
        self.lcm_thread.join()
        self.lc.unsubscribe(self.subscription)
        self.socket.close()
        del self.lc
        del self.head
        print("Thread stopped!")

    def read_states(self):
        # Get the sensor values from the shared memory
        self.head.read()
        T = self.head.get_sensor("joint_torques").copy()
        q = self.head.get_sensor("joint_positions").copy()
        dq = self.head.get_sensor("joint_velocities").copy()
        return [q, dq, T]

    def write_command(self, cmd):
        assert (
            max(cmd.shape) == 7
        ), "The control command should be a vector of 7 numbers!"
        # Write the sensor values to the shared memory
        #         self.head.set_control("ctrl_joint_torques", cmd.reshape(7,1))
        self.head.set_control("ctrl_joint_velocities", cmd.reshape(7, 1))

        self.head.set_control(
            "ctrl_stamp", np.array(self.trigger_timestamp).reshape(1, 1) / 1000000
        )
        self.head.write()
        self.cmd_log = cmd

    def generate_plot_data(self, state, cmd):
        q, dq, T = state
        data = {
            "timestamp": self.trigger_timestamp,
            "cmd": cmd.tolist(),
            "robot_states": {"q": q.tolist(), "dq": dq.tolist(), "torques": T.tolist()},
        }
        return data

    def trigger_callback(self, channel, data):
        msg = ipc_trigger_t.decode(data)
        self.trigger_timestamp = msg.timestamp
        self.trigger_counter += 1

        if self.trigger_counter % self.ds_ratio == 0:
            state = self.read_states()

            if self.controller is not None and self.pause == False:
                cmd = self.controller(
                    self.trigger_timestamp / 1000000.0, state, self.initial_states
                )
                self.write_command(cmd)
            else:
                cmd = np.zeros(7)

            if self.plotting:
                data = self.generate_plot_data(state, self.cmd_log)
                self.socket.send_string(json.dumps(data))

            if (
                self.logging
                and self.trigger_counter / self.ds_ratio < self.max_log_count
            ):
                self.log_data.append(
                    [state, self.cmd_log.copy(), self.trigger_timestamp]
                )

    def get_recorded_dataset(self):
        states = []
        for i in range(len(self.log_data[0][0])):
            states.append(np.vstack([d[0][i] for d in self.log_data]))

        cmds = np.vstack([d[1].T for d in self.log_data])
        stamps = np.vstack([d[2] for d in self.log_data])

        return stamps, states, cmds
