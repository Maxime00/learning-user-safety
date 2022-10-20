import matplotlib.pyplot as plt
import numpy as np
from learning_safety_margin.vel_control_utils import *

def plot_f0_traj(rec):
    # Plots for going to start position of trajectory

    traj_name ="Going to x0"

    # Formatting for ease
    rec_time = rec[:, 0]

    rec_pos = rec[:, 1:8]
    rec_ref_pos = rec[:, 15:22]

    rec_int = rec[:, 22:29]
    rec_int_tor = rec[:, 29:36]

    rec_vel = rec[:, 8:15]

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")

    # Plot reference and state
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint state and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(rec_time[:], rec_ref_pos[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6,.15))

    # Plot velocities and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocities of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_vel[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded'], loc=(.6, .15))

    # Plot integrator value
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint integrator of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_int[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")

    # Plot torques and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint integrator Torques of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_int_tor[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded'], loc=(.6, .15))

    plt.show()

def plot_rec_from_planned_joint(rec, traj_name):
    #########
    # Plot reference vs recording of joint controller from planned trajectory (from CBF planner)
    ########

    # Formatting for ease
    rec_time = rec[:, 0]

    rec_pos = rec[:, 1:8]
    rec_ref_pos = rec[:, 22:29]

    rec_vel = rec[:, 8:15]
    rec_ref_vel = rec[:, 29:36]

    rec_torques = rec[:, 15:22]
    rec_ref_torques = rec[:, 36:43]

    rec_cart_pos = rec[:, 43:46]
    rec_cart_vel = rec[:, 46:49]

    rec_ref_cart_pos = rec[:, 49:52]
    rec_ref_cart_vel = rec[:, 52:55]

    rec_ort = rec[:, 55:59]
    rec_ref_ort = rec[:, 59:63]

    rec_acc = rec[:, 63:66]
    rec_ref_acc = rec[:, 66:69]

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")

    # Plot reference and state
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint state and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(rec_time[:], rec_ref_pos[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6,.15))

    # Plot velocities and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocities of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_vel[:, i])
        ax.plot(rec_time, rec_ref_vel[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # Plot torques and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint  Torques of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_torques[:, i])
        ax.plot(rec_time, rec_ref_torques[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # 3D position plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_cart_pos[:, 0], rec_cart_pos[:, 1], rec_cart_pos[:, 2])
    plt.plot(rec_ref_cart_pos[:, 0], rec_ref_cart_pos[:, 1], rec_ref_cart_pos[:, 2])
    plt.xlim(ws_lim[0])
    plt.ylim(ws_lim[1])
    ax.set_zlim(ws_lim[2])
    fig.suptitle("EE pos vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # 3D velocity plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_cart_vel[:, 0], rec_cart_vel[:, 1], rec_cart_vel[:, 2])
    plt.plot(rec_ref_cart_vel[:, 0], rec_ref_cart_vel[:, 1], rec_ref_cart_vel[:, 2])
    plt.xlim(np.array(vdot_lim) / 2)
    plt.ylim(np.array(vdot_lim) / 2)
    ax.set_zlim(np.array(vdot_lim) / 2)
    fig.suptitle("EE vel vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot reference and state
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Position and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time[:], rec_cart_pos[:, i])
        ax.plot(rec_time[:], rec_ref_cart_pos[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # # Plot position error
    # fig, axs = plt.subplots(3, 1)
    # fig.suptitle("EE position error of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()):
    #     ax.plot(rec_time, error_pos[:, i])
    #     ax.set(ylabel="Dim {}".format(i))
    #     ax.set(xlabel="Time [sec]")

    # Plot velocities and reference
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Velocity and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_cart_vel[:, i])
        ax.plot(rec_time, rec_ref_cart_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot velocity error
    # fig, axs = plt.subplots(3, 1)
    # fig.suptitle("EE Velocity error of '{}'".format(traj_name))
    #
    # for i, ax in enumerate(axs.ravel()):
    #     ax.plot(rec_time, error_vel[:, i])
    #     ax.set(ylabel="Dim {}".format(i))
    #     ax.set(xlabel="Time [sec]")

    # Plot acc and reference - no acceleration recorded
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE acceleration and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_acc[:, i])
        ax.plot(rec_time, rec_ref_acc[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot orientation and reference
    fig, axs = plt.subplots(4, 1)
    fig.suptitle("EE Orientation and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_ort[:, i])
        ax.plot(rec_time, rec_ref_ort[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    plt.show()

def pandas_to_np(df):
    ## Convert pandas dataframe into numpy list for plotting
    temp_time = df['time'].to_numpy()
    temp_pos = df['position'].to_numpy()
    temp_vel = df['velocity'].to_numpy()
    temp_torques = df['torques'].to_numpy()
    reference_arr = np.zeros((len(temp_time), 1 + 7 * 3))
    reference_arr[:, 0] = temp_time
    for i in range(0, (len(temp_pos))):
        reference_arr[i, 1:8] = temp_pos[i]
        reference_arr[i, 8:15] = temp_vel[i]
        reference_arr[i, 15:22] = temp_torques[i]

    return reference_arr

def plot_rec_from_replay(df_ref, rec, traj_name):

    ref = pandas_to_np(df_ref)

    # Formatting for ease
    rec_time = rec[:, 0]
    ref_time = ref[:,0]

    rec_pos = rec[:, 1:8]
    ref_pos = ref[:, 1:8]
    rec_ref_pos = rec[:, 22:29]
    error_pos = rec_pos - rec_ref_pos

    rec_vel = rec[:, 8:15]
    ref_vel = ref[:,8:15]
    rec_ref_vel = rec[:, 29:36]
    error_vel = rec_vel - rec_ref_vel

    rec_tor = rec[:, 15:22]
    ref_tor = ref[:, 15:22]
    rec_ref_tor = rec[:, 36:43]
    error_tor = rec_tor - rec_ref_tor

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")
    print("Length of reference : ", len(ref_time), "\n Length of recorded : ", len(rec_time))

    # Plot reference and state
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint state and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(ref_time[:], ref_pos[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6,.15))

    # Plot position error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint position error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_pos[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")


    # Plot velocities and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocities and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_vel[:, i])
        ax.plot(ref_time, ref_vel[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # Plot velocity error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Velocity error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_vel[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")

    # Plot torques and reference
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_tor[:, i])
        ax.plot(ref_time, ref_tor[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'], loc=(.6, .15))

    # Plot torque error
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, error_tor[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")

    plt.show()

def plot_rec_from_planned_cart(pos_vel, acc, t, rec):
    #########
    # Plot reference vs recording of cartesian controller from planned trajectory (from CBF planner)
    ########
    traj_name ="Cartesian Controller"

    # Formatting for ease
    rec_time = rec[:, 0]
    ref_time = t

    rec_pos = rec[:, 1:4]
    ref_pos = pos_vel[:, 0:3]
    rec_ref_pos = rec[:, 10:13]
    error_pos = rec_pos - rec_ref_pos

    rec_vel = rec[:, 4:7]
    ref_vel = pos_vel[:, 3:6]
    rec_ref_vel = rec[:, 13:16]
    error_vel = rec_vel - rec_ref_vel

    rec_acc = rec[:, 7:10]
    ref_acc = acc
    rec_ref_acc = rec[:, 16:19]
    error_acc = rec_acc - rec_ref_acc

    rec_ort = rec[:, 26:30]
    rec_ref_ort = rec[:, 30:34]

    rec_ang_vel = rec[:, 34:37]
    rec_ref_ang_vel = rec[:, 37:40]

    rec_tor = rec[:, 19:26]

    print("PLOTTING REFERENCE VS RECORDED TRAJECTORY")
    print("Length of reference : ", len(ref_time), "\n Length of recorded : ", len(rec_time))

    # 3D position plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_pos[:, 0], rec_pos[:, 1], rec_pos[:, 2])
    plt.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2])
    plt.xlim(ws_lim[0])
    plt.ylim(ws_lim[1])
    ax.set_zlim(ws_lim[2])
    fig.suptitle("EE pos vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # 3D velocity plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.plot(rec_vel[:, 0], rec_vel[:, 1], rec_vel[:, 2])
    plt.plot(ref_vel[:, 0], ref_vel[:, 1], ref_vel[:, 2])
    plt.xlim(np.array(vdot_lim)/2)
    plt.ylim(np.array(vdot_lim)/2)
    ax.set_zlim(np.array(vdot_lim)/2)
    fig.suptitle("EE vel vs ref of '{}'".format(traj_name))
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot reference and state
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Position and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time[:], rec_pos[:, i])
        ax.plot(ref_time[:], ref_pos[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot position error
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE position error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, error_pos[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")

    # Plot velocities and reference
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Velocity and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_vel[:, i])
        ax.plot(ref_time, ref_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot velocity error
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE Velocity error of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, error_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")

    # Plot acc and reference - no acceleration recorded
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE acceleration and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_acc[:, i])
        ax.plot(ref_time, ref_acc[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot orientation and reference
    fig, axs = plt.subplots(4, 1)
    fig.suptitle("EE Orientation and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_ort[:, i])
        ax.plot(rec_time, rec_ref_ort[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot angular velocity and reference
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("EE angular velocity and reference of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()):
        ax.plot(rec_time, rec_ang_vel[:, i])
        ax.plot(rec_time, rec_ref_ang_vel[:, i])
        ax.set(ylabel="Dim {}".format(i))
        ax.set(xlabel="Time [sec]")
    fig.legend(labels=['Recorded', 'Reference'])

    # Plot torques
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Joint Torques of '{}'".format(traj_name))

    for i, ax in enumerate(axs.ravel()[:-1]):
        ax.plot(rec_time, rec_tor[:, i])
        ax.set(ylabel="joint {}".format(i+1))
        ax.set(xlabel="Time [sec]")

    plt.show()
