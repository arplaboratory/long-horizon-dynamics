import pandas as pd
from pyulog import ULog
from matplotlib import pyplot as plt

# plt.rcParams["figure.figsize"] = (20, 16)
# plt.rcParams['lines.linewidth'] = 6
# plt.rcParams.update({
#     #"text.usetex": True,
#     "font.family": "sans",
#     "font.size": 24,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42
# })
# colors = ["#365282", "#9FBDA0","#edb120", "#E84C53", "#ff7f00", "#984ea3", "#f781bf", "#999999"]

class ProcessULog:

    def __init__(self, ulog_file, messages):
        self.ulog_file = ulog_file
        self.messages = messages
        self.ulog = ULog(ulog_file)

    def get_data(self):
        data = {}
        for m in self.messages:
            try:
                dataset = self.ulog.get_dataset(m)
                data[m] = dataset.data
            except KeyError:
                print(f"Message '{m}' not found in the ULog file.")
        return data
    
    def get_df(self):
        data = self.get_data()
        df = {}
        for m in self.messages:
            df[m] = pd.DataFrame(data[m])
        return df

    def sync_df(self, df_list):
        df = pd.concat(df_list, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns
        df = df.drop_duplicates(subset='timestamp', keep='first')
        df = df.set_index('timestamp')
        return df
    
    def get_sync_df(self):
        df_list = [df for df in self.get_df().values() if not df.empty]
        return self.sync_df(df_list)
    
    def save_csv(self, df, file_name):
        df.to_csv(file_name)
    

if __name__ == '__main__':

    ulog_file = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/arpl_fixed/px4_ulog/log_91_2024-5-31-12-44-56.ulg'
    messages = ['actuator_outputs', 'actuator_outputs', 'actuator_servos', 'vehicle_local_position', 'vehicle_attitude', 'vehicle_angular_velocity']
    process = ProcessULog(ulog_file, messages)
    df = process.get_sync_df()
    # process.save_csv(df, 'output.csv')

    # Plot the following headers
    headers = ['control[0]', 'control[1]', 'control[2]', 'control[3]', 'q[0]', 'q[1]', 'q[2]', 'q[3]', 'xyz[0]', 'xyz[1]', 'xyz[2]', 'vx', 'vy', 'vz']

    # Create subplots for control, attitude, angular velocity and velocity
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    

    # Control 
    for i in range(4):
        axs[0].plot(df.index, df[headers[i]], label=headers[i])
    
    axs[0].set_title('Control')
    axs[0].legend()
    axs[0].grid()

    # Attitude
    for i in range(4, 8):
        axs[1].plot(df.index, df[headers[i]], label=headers[i])
        
    axs[1].set_title('Attitude')
    axs[1].legend()

    # Angular Velocity
    for i in range(8, 11):
        axs[2].plot(df.index, df[headers[i]], label=headers[i])

    axs[2].set_title('Angular Velocity')
    axs[2].legend()
    
    # Velocity
    for i in range(11, 14):
        axs[3].plot(df.index, df[headers[i]], label=headers[i])

    axs[3].set_title('Velocity')
    axs[3].legend()

    plt.show()
