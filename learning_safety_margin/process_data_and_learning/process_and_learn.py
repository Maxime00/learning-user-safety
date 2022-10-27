import os
import argparse
from bag2csv import process_user_rosbags
from learning_cbf_pos_lim import pos_learning
from learning_cbf_vel_lim import vel_learning

def set_up_dir(user_number):
    # Check directories and create them if needed
    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number+"/csv"

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(os.path.join(data_dir, "daring")):
        os.mkdir(os.path.join(data_dir, "daring"))

    if not os.path.isdir(os.path.join(data_dir, "safe")):
        os.mkdir(os.path.join(data_dir, "safe"))

    if not os.path.isdir(os.path.join(data_dir, "unsafe")):
        os.mkdir(os.path.join(data_dir, "unsafe"))

def bags_need_processing(user_number):

    data_dir = "/home/ros/ros_ws/src/learning_safety_margin/data/User_"+user_number

    count_csv = sum(len(files) for _, _, files in os.walk(data_dir + "/csv"))
    count_rosbags = sum(len(files) for _, _, files in os.walk(data_dir + "/rosbags"))

    print(f"# of rosbags : {count_rosbags} \t # of csv : {count_csv}")

    # bag2csv outputs 5 files per bag
    if count_csv >= 5*count_rosbags:
        return False
    elif count_csv < 5*count_rosbags:
        return True

def process_and_learn(user_number, learning_algorithm ='vel', to_smooth = '1'):
    set_up_dir(user_number)

    if bags_need_processing(user_number):
        print("Processing rosbags for User_" + user_number)
        process_user_rosbags(user_number, to_smooth)
        print("Finished processing rosbags for User_" + user_number)
    else:
        print("Rosbags already processed for User_"+ user_number)

    if learning_algorithm == 'pos':
        print("Learning position limits cbf for User_" + user_number)
        pos_learning(user_number)
    if learning_algorithm == 'vel':
        print("Learning velocity limits cbf for User_" + user_number)
        vel_learning(user_number)

def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Process rosbags into csv and runs cbf learning")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default=False)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-u", "--user_number", type=str, default='0', nargs='?', help="The user data to process")
    parser.add_argument("-l", "--learning_algorithm", type=str, choices=['pos', 'vel'], default='vel',  nargs='?', help="The learning algorithm to run")
    parser.add_argument("-s", "--to_smooth", type=str, choices=['0', '1'], default='1', nargs='?', help="Option to smooth cartesian velocities")

    args = parser.parse_args()

    return args.user_number, args.learning_algorithm, args.to_smooth

if __name__ == "__main__":
    user_nbr, learning_alg, smooth_flag = parser()
    process_and_learn(user_nbr, learning_alg, smooth_flag)
