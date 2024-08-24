from utils import DataPreprocessing

config_file = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/configs/fixedwing.yaml'
ulog_path = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/arpl_fixed/px4_ulog/log_91_2024-5-31-12-44-56.ulg'

save_csv_path = '/home/pratyaksh/arpl/workspaces/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/arpl_fixed/px4_ulog/pre_processed.csv'
data_preprocessor = DataPreprocessing(config_file=config_file, selection_var="none")
data_preprocessor.loadLogs(ulog_path)
data_df = data_preprocessor.get_dataframes()

# Visualize the data
data_preprocessor.visualize_data()

# data_df.to_csv(save_csv_path, index=False)


    