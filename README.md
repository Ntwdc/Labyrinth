# Labyrinth
PID controller and MPC controller for a Labyrinth robot

Software: Python 3.9.12, Spyder 5.3.1, java 1.8.0_333, Netbeans 8.2, jAER 2022.07.08

Hardware: Intel 12700-H, GTX 3060Ti, DAVIS240C, Arduino MKRZERO


![1663168306363](https://user-images.githubusercontent.com/39051034/190193648-ea5e93a3-8006-4c22-a607-1b0c329b32e6.jpg)



Description of files:

Labi_MPC.py: implementation of a mpc controller

MPC.py: a class for mpc

Labi_PID.py: implementation of a PID controller

PID.py: a class for PID from https://github.com/m-lundberg/simple-pid

CNN_ER_FA.py: CNN from https://github.com/FabianAmherd/CNN_ER_FA

labiutils.py: some methods used used in the script

globals_and_utils.py: some methods used from https://github.com/SensorsINI/joker-network

params.py: a class managing the ball's information

servo_test.py: a test of servo control

labi-test.ino: arduino script

big_ds_1_image.pt and big_ds_3_image.pt: network state-dicts from https://github.com/FabianAmherd/CNN_ER_FA

warm.pt: some data for warmup

path.npy: the coordinates of some points on the path
