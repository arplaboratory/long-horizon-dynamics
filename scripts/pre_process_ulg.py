from utils import DataPreprocessing
import os 
import sys
from dynamics_learning.utils import check_folder_paths



# Set global paths 
folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
resources_path = folder_path + "resources/"
configs_path = folder_path + "configs/"

config_file = configs_path + "fixedwing.yaml"
ulog_dir = resources_path + "data/arpl_fixed/px4_ulog/"
save_csv_dir = resources_path + "data/arpl_fixed/px4_csv/"

# check folder paths
check_folder_paths([ulog_dir, save_csv_dir])

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