"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import torch
from PIL import Image
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
from examples.viewer import Viewer
from examples.Event_camera_sim import Event_simulator
from examples.Helper import Helper
from examples.Train_Network import Train

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=True,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=3*1000,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    ### Init viewer tool#######################################
    view_tool = Viewer()
    Event_creator = Event_simulator()
    Helper = Helper()
    Train_net = Train()
    Train_flag=1
    predict_flag=0
    #### Initialize the simulation #############################
    frame_dimensions=[64,64]
    Previous_frame = np.zeros((frame_dimensions[0], frame_dimensions[1], 3),dtype=np.uint8)
    events_sequence=torch.zeros((1,  1,frame_dimensions[0], frame_dimensions[1]))
    sequence_length=20
    H = .3
    H_STEP = .1
    R = .5
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(ARGS.num_drones)])
    Angular_velocities=[]
  #  INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])

    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])



    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()

    def create_inference(sim_time,f=0.1,pitch_angle=0):

        sin_wave=5*np.sin(5*np.pi*sim_time*f)#sine wave amplitue 0.3
        pitch_angle=pitch_angle+sin_wave*env.SIM_FREQ
        # print('sine wave',sin_wave)
        return sin_wave,pitch_angle
    epochs=Train_net.num_epochs


    epoch=0
    model = Helper.LoadMyModel()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        if i%int(ARGS.duration_sec*env.SIM_FREQ/epochs)==0 and i!=0 :
            epoch=epoch+1
            print(epoch)
        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
        sine_wave_pitch,pitch_angle=create_inference(i*epochs/(ARGS.duration_sec*env.SIM_FREQ))
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        angular_velocities=Helper.Get_angular_velocities(obs)
        # view_tool.RGB_view(obs)
        Current_frame=view_tool.RGB_get(obs)
        Event_creator.Event_From_Adjecent_Frames(Current_frame,Previous_frame)
        # view_tool.Event_view(Event_creator.pos_event,Event_creator.neg_event,event_frame=Event_creator.event_frame)

        #### Compute control at the desired frequency ##############

        Angular_velocities.append(angular_velocities)
        # model=Helper.LoadMyModel()
        # model.eval()
        if len(Angular_velocities)>sequence_length and predict_flag:
            events_sequence = events_sequence[1:, :, :,:]
            predict=model.forward(torch.tensor(torch.tensor(events_sequence).float()))
            print('error',predict[0]-Angular_velocities[-1][2])
            print('predicted yaw',predict[0],'yaw ground truth',Angular_velocities[-1][2])
        if len(Angular_velocities)>sequence_length and Train_flag and i>30:
            events_sequence=events_sequence[1:,:,:,:]
            Angular_velocities=Angular_velocities[1:]
            Train_net.train(Train_net.model.float(),torch.tensor(events_sequence).float(),torch.tensor(Angular_velocities).float(),epoch)
        new_event=torch.unsqueeze(torch.unsqueeze(torch.tensor(Event_creator.Event_From_Adjecent_Frames(Current_frame,Previous_frame)),dim=0),dim=0)
        events_sequence = np.concatenate((events_sequence,new_event ), axis=0)
        Previous_frame = Current_frame
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=[0,0,0.5],#np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                       # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                       target_rpy=[0,0,sine_wave_pitch] ,#INIT_RPYS[j, :]
                                                                       target_rpy_rates=[0,0,0.5]
                                                                       )

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0



        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            # if ARGS.vision:
            #     for j in range(ARGS.num_drones):
            #         print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
            #               obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
            #               obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
            #               )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)
        if i%int(ARGS.duration_sec*env.SIM_FREQ/epochs)==0 and epoch > 0 and Train_flag :
            print("Epoch: %d, loss: %1.5f" , epoch)
            print('predict accuracy', Train_net.train_accuracy_MSE[-1])
            print('predicted value',Train_net.predicted,'GroundTruth',Train_net.GroundTruth)
            print('loss value', Train_net.Loss_history[-1])
            Helper.SavemyModel(Train_net.model)
            # plt.figure()
            # plt.plot(Train_net.epochs, Train_net.Loss_history, 'g', label='Training loss')
            # # plt.plot(epochs, Train_net.GroundTruth, 'b', label='Ground Truth')
            # plt.title('Training and Validation loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show(block=False)
            # plt.pause(0.1)
    #### Close the environment #################################
    env.close()


