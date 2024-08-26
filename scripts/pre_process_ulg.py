from utils import DataPreprocessing
import os 

config_file = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/configs/fixedwing.yaml'
ulog_dir = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/arpl_fixed/px4_ulog/'
save_csv_dir = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/arpl_fixed/px4_csv/'
data_preprocessor = DataPreprocessing(config_file=config_file, selection_var="none")

# load logs for all ulog files in the specified directory 

# Loop through all ulog files in the specified directory and get the absolute path of each file
ulog_files = [f for f in os.listdir(ulog_dir) if f.endswith('.ulg')]

for ulog_file in ulog_files:
    ulog_path = os.path.join(ulog_dir, ulog_file)
    data_preprocessor.loadLogs(ulog_path)
    data_df = data_preprocessor.get_dataframes()
    save_csv_path = os.path.join(save_csv_dir, ulog_file.split('.')[0] + '.csv')
    data_preprocessor.select_data(save_csv_path)