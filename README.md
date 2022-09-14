# Labyrinth
PID controller and MPC controller for a Labyrinth robot

Software: Python 3.9.12, Spyder 5.3.1, java 1.8.0_333, Netbeans 8.2, jAER 2022.07.08

Hardware: Intel 12700-H, GTX 3060Ti, DAVIS240C, Arduino MKRZERO



How to use:

Connect the electrical circuit and upload labi-test.ino to the arduino board.

Download jAER and run it in Netbeans. Enable the UdpFramer filter in jAER and set the port number and the buffer size.   

Run Labi_MPC.py or Labi_PID.py in python. During running, press ctrl C to stop the program and get the plot.

Some variables that need to be set are list at the beginning of Labi_MPC.py and Labi_PID.py. 

If a new camera is used, a recalibration is needed and the camera parameters should be modified.

Because the input of CNN is three 240*180 images, the image size needs to be adjusted if a different resolution is used. 


![1663169806195](https://user-images.githubusercontent.com/39051034/190199460-be2bd4d7-7538-4db8-8d43-c49f8973d777.jpg)


Description of files:

Labi_MPC.py: implementation of a mpc controller

MPC.py: a class for mpc

Labi_PID.py: implementation of a PID controller

PID.py: a class for PID from https://github.com/m-lundberg/simple-pid

CNN_ER_FA.py: CNN from https://github.com/FabianAmherd/CNN_ER_FA

labiutils.py: some methods used in the script

globals_and_utils.py: some methods used from https://github.com/SensorsINI/joker-network

params.py: a class managing the ball's information

servo_test.py: a test of servo control

labi-test.ino: arduino script

big_ds_1_image.pt and big_ds_3_image.pt: network state-dicts from https://github.com/FabianAmherd/CNN_ER_FA

warm.pt: some data for warmup

path.npy: the coordinates of some points on the path
