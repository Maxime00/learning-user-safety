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

    if len(os.listdir(data_dir + "/csv")) >= 3*len(os.listdir(data_dir + "/rosbags")):
        return False
    elif len(os.listdir(data_dir + "/csv")) < 3*len(os.listdir(data_dir + "/rosbags")):
        return True


def parser():
    # Parse arguments calls functions accordingly

    parser = argparse.ArgumentParser(description="Process rosbags into csv and runs cbf learning")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=True)
    group.add_argument("-q", "--quiet", action="store_true", default = False)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-u", "--user_number", type=str, default='0', nargs='?', help="The user data to process")
    parser.add_argument("-l", "--learning_algorithm", type=str, choices=['pos', 'vel'], default = 'pos',  nargs='?', help="The learning algorithm to run")

    args = parser.parse_args()

    set_up_dir(args.user_number)

    if bags_need_processing():
        print("Processing rosbags for User_" + args.user_number)
        process_user_rosbags(args.user_number)
        print("Finished processing rosbags for User_" + args.user_number)
    else :
        print("Rosbags already processed for User_"+ args.user_number)

    if args.learning_algorithm == 'pos':
        print("Learning position limits cbf for User_" + args.user_number)
        pos_learning(args.user_number)
    if args.learning_algorithm == 'vel':
        print("Learning velocity limits cbf for User_" + args.user_number)
        vel_learning(args.user_number)

if __name__ == "__main__":
    parser()

