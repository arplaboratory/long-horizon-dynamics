import rospy
import csv 
import genpy
import pandas as pd
import bagpy
import numpy as np

def parseBag(topic, path):
  bag = bagpy.bagreader(path, verbose=False)
  return pd.read_csv(bag.message_by_topic(topic))

def extract_topic_data(bag_file_path, output_csv_file):
   
    df_odom = parseBag('/apm/mavros/odometry/in', bag_file_path)

    # Compute time 
    t_odom = df_odom.apply(lambda r: rospy.Time(r["header.stamp.secs"], r["header.stamp.nsecs"]).to_sec(), axis=1)

    # Position
    p = df_odom[["pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z"]].to_numpy()
   
    # Velocity
    v = df_odom[["twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z"]].to_numpy()

    # Attitude
    q = df_odom[["pose.pose.orientation.x", "pose.pose.orientation.y", "pose.pose.orientation.z", "pose.pose.orientation.w"]].to_numpy()

    # Angular velocity
    w = df_odom[["twist.twist.angular.x", "twist.twist.angular.y", "twist.twist.angular.z"]].to_numpy()

    # Save to CSV
    df = pd.DataFrame(data=np.hstack((p, v, q, w)), columns=["px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "wx", "wy", "wz"])
    df.insert(0, "t", t_odom)
    df.to_csv(output_csv_file, index=False)



if __name__ == "__main__":
    bag_file_path = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/rosbags/og.bag'
    topic_name_to_extract = '/apm/mavros/odometry/in'
    output_csv_file = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/rosbags/og.csv'

    extract_topic_data(bag_file_path, output_csv_file)
