import pandas as pd
import math
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from pyulog import ULog
from utils.model_config import ModelConfig
from utils.ulog_tools import load_ulog, pandas_from_topic
from utils.dataframe_tools import compute_flight_time, resample_dataframe_list
from utils.quat_utils import quaternion_to_rotation_matrix
from progress.bar import Bar
from matplotlib.backends.backend_pdf import PdfPages
import vpselector 

plt.rcParams["figure.figsize"] = (20, 16)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "sans",
    "font.size": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})
colors = ["#365282", "#9FBDA0","#edb120", "#E84C53", "#ff7f00", "#984ea3", "#f781bf", "#999999"]

class DataPreprocessing(object):

    visual_dataframe_selector_config_dict = {
        "x_axis_col": "timestamp",
        "sub_plt1_data": ["q0", "q1", "q2", "q3"],
        "sub_plt2_data": ["throttle", "aileron", "elevator", "rudder"],
        # "sub_plt3_data": ["vx", "vy", "vz"],
        "sub_plt4_data": ["ang_vel_x", "ang_vel_y", "ang_vel_z"],
        # plot wind speed
        # "sub_plt5_data": ["wind_north", "wind_east"],
        # "sub_plt6_data": ["x", "y", "z"],
    }

    def __init__(self, config_file, selection_var="none"):
        
        self.config = ModelConfig(config_file)
        config_dict = self.config.dynamics_model_config

        assert type(config_dict) is dict, "req_topics_dict input must be a dict"
        assert bool(config_dict), "req_topics_dict can not be empty"
        self.config_dict = config_dict
        self.resample_freq = config_dict["resample_freq"]
        self.estimate_angular_acceleration = config_dict[
            "estimate_angular_acceleration"
        ]
        print("Resample frequency: ", self.resample_freq, "Hz")
        self.req_topics_dict = config_dict["data"]["required_ulog_topics"]

        if selection_var != "none":
            split = selection_var.split("/")
            assert (
                len(split) == 2
            ), "Setpoint variable must be of the form: topic_name/variable_name"

            topic_name = split[0]
            variable_name = split[1]

            if topic_name in self.req_topics_dict.keys():
                assert (
                    "ulog_name" in self.req_topics_dict[topic_name].keys()
                    and "dataframe_name" in self.req_topics_dict[topic_name].keys()
                ), "Topic already exists but does not have the required keys"

                if variable_name in self.req_topics_dict[topic_name]["ulog_name"]:
                    raise AttributeError(
                        "Please only variables for setpoint data selection that are not used in a different context for system identification"
                    )

                assert (
                    "timestamp" in self.req_topics_dict[topic_name]["ulog_name"]
                    and "timestamp"
                    in self.req_topics_dict[topic_name]["dataframe_name"]
                ), "Topic already exists but does not have the required timestamp key"

                self.req_topics_dict[topic_name]["ulog_name"].append(variable_name)
                self.req_topics_dict[topic_name]["dataframe_name"].append(variable_name)

            else:
                self.req_topics_dict[topic_name] = {
                    "ulog_name": ["timestamp", variable_name],
                    "dataframe_name": ["timestamp", variable_name],
                }

            print(
                "Augmented required topics list with setpoint variable:",
                variable_name,
                "from topic",
                topic_name,
            )

        self.req_dataframe_topic_list = config_dict["data"]["req_dataframe_topic_list"]

        # used to generate a dict with the resulting coefficients later on.
        self.coef_name_list = []
        self.result_dict = {}

    def loadLogs(self, rel_data_path):
        self.rel_data_path = rel_data_path
        self.loadLogFile(rel_data_path)
        
    def loadLogFile(self, rel_data_path):
        if rel_data_path.endswith(".csv"):
            print("Loading CSV file: ", rel_data_path)
            self.data_df = pd.read_csv(rel_data_path, index_col=0)
            print("Loading topics: ", self.req_dataframe_topic_list)
            for req_topic in self.req_dataframe_topic_list:
                assert req_topic in self.data_df, "missing topic in loaded csv: " + str(
                    req_topic
                )
            return True

        elif rel_data_path.endswith(".ulg"):
            print("Loading uLog file: ", rel_data_path)
            ulog = load_ulog(rel_data_path)
            print("Loading topics:")
            for req_topic in self.req_topics_dict:
                print(req_topic)
            self.check_ulog_for_req_topics(ulog)

            # compute flight time based on the landed topic
            landed_df = pandas_from_topic(ulog, ["vehicle_land_detected"])
            fts = compute_flight_time(landed_df)

            if len(fts) == 1:
                self.data_df = self.compute_resampled_dataframe(ulog, fts[0])
            else:
                for ft in fts:
                    # check if the dataframe already exists and if so, append to it
                    if getattr(self, "data_df", None) is None:
                        self.data_df = self.compute_resampled_dataframe(ulog, ft)
                    else:
                        self.data_df.append(self.compute_resampled_dataframe(ulog, ft))

            return True

        else:

            return False
    
    def check_ulog_for_req_topics(self, ulog):
        for topic_type in self.req_topics_dict.keys():
            try:
                topic_dict = self.req_topics_dict[topic_type]
                if "id" in topic_dict.keys():
                    id = topic_dict["id"]
                    topic_type_data = ulog.get_dataset(topic_type)
                else:
                    topic_type_data = ulog.get_dataset(topic_type)
            except:
                print("Missing topic type: ", topic_type)
                exit(1)
            topic_type_data = topic_type_data.data
            ulog_topic_list = self.req_topics_dict[topic_type]["ulog_name"]
            for topic_index in range(len(ulog_topic_list)):
                try:
                    topic = ulog_topic_list[topic_index]
                    topic_data = topic_type_data[topic]
                except:
                    print("Missing topic: ", topic_type, ulog_topic_list[topic_index])
                    exit(1)
        return
    
    def compute_resampled_dataframe(self, ulog, fts):
        print("Starting data resampling of topic types: ", self.req_topics_dict.keys())
        # setup object to crop dataframes for flight data
        df_list = []
        topic_type_bar = Bar("Resampling", max=len(self.req_topics_dict.keys()))

        # getting data
        for topic_type in self.req_topics_dict.keys():
            topic_dict = self.req_topics_dict[topic_type]

            if "id" in topic_dict.keys():
                id = topic_dict["id"]
                curr_df = pandas_from_topic(ulog, [topic_type], id)
            else:

                curr_df = pandas_from_topic(ulog, [topic_type])

            curr_df = curr_df[topic_dict["ulog_name"]]
            if "dataframe_name" in topic_dict.keys():
                assert len(topic_dict["dataframe_name"]) == len(
                    topic_dict["ulog_name"]
                ), (
                    "could not rename topics of type",
                    topic_type,
                    "due to rename list not having an entry for every topic.",
                )
                curr_df.columns = topic_dict["dataframe_name"]
            topic_type_bar.next()
            if (
                topic_type == "vehicle_angular_velocity"
                and self.estimate_angular_acceleration
            ):
                ang_vel_mat = curr_df[
                    ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
                ].to_numpy()
                time_in_secods_np = curr_df[["timestamp"]].to_numpy() / 1000000
                time_in_secods_np = time_in_secods_np.flatten()
                ang_acc_np = np.gradient(ang_vel_mat, time_in_secods_np, axis=0)
                topic_type_bar.next()
                curr_df[["ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]] = ang_acc_np

            df_list.append(curr_df)

        topic_type_bar.finish()

        # Check if actuator topics are empty
        if not fts:
            print("could not select flight time due to missing actuator topic")
            exit(1)

        if isinstance(fts, list):
            resampled_df = []
            for ft in fts:
                new_resampled_df = resample_dataframe_list(
                    df_list, ft, self.resample_freq, slerp_enabled=True
                )
                resampled_df.append(new_resampled_df)
            resampled_df = pd.concat(resampled_df, ignore_index=True)
        else:
            resampled_df = resample_dataframe_list(df_list, fts, self.resample_freq)

        if self.estimate_angular_acceleration:
            ang_vel_mat = resampled_df[
                ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
            ].to_numpy()
            for i in range(3):
                ang_vel_mat[:, i] = (
                    np.convolve(ang_vel_mat[:, i], np.ones(33), mode="same") / 33
                )

        
            time_in_secods_np = resampled_df[["timestamp"]].to_numpy() / 1000000
            time_in_secods_np = time_in_secods_np.flatten()
            ang_acc_np = np.gradient(ang_vel_mat, time_in_secods_np, axis=0)
            topic_type_bar.next()
            resampled_df[["ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]] = ang_acc_np

        return resampled_df.dropna()

    def get_dataframes(self):
        return self.data_df


    def visualize_data(self):
        
        position_headers = ["x", "y", "z"]
        velocity_headers = ["vx", "vy", "vz"]
        acceleration_headers = ["acc_b_x", "acc_b_y", "acc_b_z"]
        angular_velocity_headers = ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        quaternion_headers = ["q0", "q1", "q2", "q3"]
        control_headers = ["throttle", "aileron", "elevator", "rudder"]
        wind_speed_headers = ["wind_north", "wind_east"]
        airspeed_headers = ["airspeed"]
        diff_pressure_headers = ["diff_pressure"]

        velocity_labels = [r'$v_x$ $[m/s]$', r'$v_y$ $[m/s]$', r'$v_z$ $[m/s]$']
        acceleration_labels = [r'$a_x$ $[m/s^2]$', r'$a_y$ $[m/s^2]$', r'$a_z$ $[m/s^2]$']
        angular_velocity_labels = [r'$\omega_x$ $[rad/s]$', r'$\omega_y$ $[rad/s]$', r'$\omega_z$ $[rad/s]$']
        quaternion_labels = [r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$']
        control_labels = [r'$\delta_{throttle}$', r'$\delta_{aileron}$', r'$\delta_{elevator}$', r'$\delta_{rudder}$']
        wind_speed_labels = [r'$v_{wind,n}$ $[m/s]$', r'$v_{wind,e}$ $[m/s]$']
        airspeed_labels = [r'$v_{airspeed}$ $[m/s]$']
        diff_pressure_labels = [r'$\Delta P$ $[Pa]$']


        # plot each of the dataframes separately and save the plots as a pdf

        # Convert timestamp to seconds and start from 0
        self.data_df["t"] = (self.data_df["t"] - self.data_df["t"].iloc[0]) / 1e6
        num_samples = self.data_df.shape[0]

        x_samples = np.arange(num_samples)
        # Local velocity
        fig, axs = plt.subplots(3, 1, figsize=(20, 16)) 
        for i in range(3):
            axs[i].plot(x_samples, self.data_df[velocity_headers[i]], color=colors[i])
            axs[i].set_ylabel(velocity_headers[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)
        axs[0].set_title("Local Velocity")
        
        # create a folder called plots in the dataset directory
        plots_path = os.path.join(os.path.dirname(os.path.abspath(self.rel_data_path)), "plots")

        print("Saving plots to: ", plots_path)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        # save all the plots as multiple pages in a single pdf

        # Creating plot file name using the name of the ulog file
        plot_file_name = os.path.basename(self.rel_data_path).split(".")[0] + "_plots.pdf"
        plot_file_path = os.path.join(plots_path, plot_file_name)
        pdf = PdfPages(plot_file_path)

        # Convert timestamp to seconds and start from 0
        self.data_df["t"] = (self.data_df["t"] - self.data_df["t"].iloc[0]) / 1e6

        # Position x-y and z in separate plots
        fig, axs = plt.subplots(2, 1, figsize=(20, 16))
        axs[0].plot(self.data_df["y"], self.data_df["x"], color=colors[0])
        axs[0].set_ylabel(r'$p_x$ $[m]$')
        axs[0].set_xlabel(r'$p_y$ $[m]$')
        axs[0].grid(alpha=0.3)
        axs[0].set_title("Trajectory")
        axs[1].plot(self.data_df["t"], -self.data_df["z"], color=colors[0])
        axs[1].set_ylabel(r'$p_z$ $[m]$')
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(alpha=0.3)

        pdf.savefig(fig)
        # Local velocity
        fig, axs = plt.subplots(3, 1, figsize=(20, 16))
        for i in range(3):
            axs[i].plot(x_samples, self.data_df[velocity_headers[i]], color=colors[i])
            axs[i].set_ylabel(velocity_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Local Velocity")
        plt.tight_layout()
        pdf.savefig(fig)

        # Local acceleration
        fig, axs = plt.subplots(3, 1, figsize=(20, 16))
        for i in range(3):
            axs[i].plot(x_samples, self.data_df[acceleration_headers[i]], color=colors[i])
            axs[i].set_ylabel(acceleration_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Local Acceleration")
        plt.tight_layout()

        pdf.savefig(fig)

        # Angular velocity
        fig, axs = plt.subplots(3, 1, figsize=(20, 16))
        for i in range(3):
            axs[i].plot(x_samples, self.data_df[angular_velocity_headers[i]], color=colors[i])
            axs[i].set_ylabel(angular_velocity_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Angular Velocity")
        plt.tight_layout()

        pdf.savefig(fig)

        # Quaternion
        fig, axs = plt.subplots(4, 1, figsize=(20, 16))
        for i in range(4):
            axs[i].plot(x_samples, self.data_df[quaternion_headers[i]], color=colors[i])
            axs[i].set_ylabel(quaternion_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Quaternion")
        plt.tight_layout()

        pdf.savefig(fig)

        # Control
        fig, axs = plt.subplots(4, 1, figsize=(20, 16))
        for i in range(4):
            axs[i].plot(x_samples, self.data_df[control_headers[i]], color=colors[i])
            axs[i].set_ylabel(control_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Control")
        plt.tight_layout()

        pdf.savefig(fig)

        # wind speed
        fig, axs = plt.subplots(2, 1, figsize=(20, 16))
        for i in range(2):
            axs[i].plot(x_samples, self.data_df[wind_speed_headers[i]], color=colors[i])
            axs[i].set_ylabel(wind_speed_labels[i])
            axs[i].set_xlabel("Samples")
            axs[i].grid(alpha=0.3)

        axs[0].set_title("Wind Speed")
        plt.tight_layout()

        pdf.savefig(fig)

        # airspeed
        fig, axs = plt.subplots(1, 1, figsize=(20, 16))
        axs.plot(x_samples, self.data_df[airspeed_headers[0]], color=colors[0])
        axs.set_ylabel(airspeed_labels[0])
        axs.set_xlabel("Samples")
        axs.grid(alpha=0.3)

        axs.set_title("Airspeed")
        plt.tight_layout()

        pdf.savefig(fig)

        # diff pressure
        fig, axs = plt.subplots(1, 1, figsize=(20, 16))
        axs.plot(x_samples, self.data_df[diff_pressure_headers[0]], color=colors[0])
        axs.set_ylabel(diff_pressure_labels[0])
        axs.set_xlabel("Samples")
        axs.grid(alpha=0.3)

        axs.set_title("Diff Pressure")
        plt.tight_layout()

        pdf.savefig(fig)
        pdf.close()    

        
    def select_data(self, save_path):
        
        # Interactive data selection using vpselector
        selected_df = vpselector.select_visual_data(self.data_df, self.visual_dataframe_selector_config_dict)

        # # Rename timestamp header to 't'
        selected_df.rename(columns={"timestamp": "t"}, inplace=True)
        selected_df.to_csv(save_path, index=False)

        self.data_df.rename(columns={"timestamp": "t"}, inplace=True)
        self.data_df = selected_df
        # self.data_df.to_csv(save_path, index=False)
    
        # plot x, y
        # fig, axs = plt.subplots(1, 1, figsize=(20, 16)) 
        # axs.plot(selected_df["y"], selected_df["x"], color=colors[0])
        # axs.set_ylabel("y")
        # axs.set_xlabel("x")
        # axs.grid(alpha=0.3)
        # axs.set_title("Trajectory")
        # # show
        # plt.show()

        