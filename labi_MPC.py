import javaobj
import serial
import socket
import cv2 
import time
import math
import torch
import queue 
import threading
import numpy as np
import matplotlib.pyplot as plt
from CNN_ER_FA import CNN
from globals_and_utils import Timer
from numba import jit
from params import Runparams
import labiutils
from MPC import MPC
#%%
x_direction = -1
y_direction = 1
width = 240
height = 180
port = 7777
buf_size = 63000
host = 'localhost'
three_images = True
isPath = True
file = "path.npy"
mtx = np.array([[250.9973, 0, 124.8801], [0, 250.7764, 62.5436], [0, 0, 1.0000]])
dist = np.array([-0.3009, 0.1034, 0, 0, 0])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (240, 180), 1, (240, 180))

#%%
@jit(nopython=True)
def transform(array, channel = 0):
    frame = np.ones(array.shape, dtype = array.dtype)
    h, w = frame.shape
    for y in range(h):
        for x in range(w):
            ry = h - y - 1
            if channel == 0:
                frame[y, x] = array[ry, x] & 0XFF
            else:
                frame[y, x] = array[ry, x] + 128
    return frame
#%%
def undistort(img):
    is_distort = False
    if is_distort:
        return img
    img = img.astype(np.uint8)
    dst = cv2.undistort(img, mtx, dist, None, mtx) 
    return dst  
#%%
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x,y])
    if event == cv2.EVENT_RBUTTONDOWN:  
        pass
    if event == cv2.EVENT_MBUTTONDOWN:
        pass
#%%
def comp(x,y):
    if y == 0 and x == 0:
        return 0, 0
    else:
        return x/(x + y), y/(x + y)
#%%
def control():
    x0 = 0
    y0 = 0
    x_hori = 84
    y_hori = 86
    accumulationx = 0.0
    accumulationy = 0.0
    tilt_x = x_hori
    tilt_y = y_hori
    params = Runparams(0.29, 0.245)
    is_pre = True
    
    print("control thread starts")
    arduino = serial.Serial('COM3', 115200, timeout = 5)
    print("arduino connected") # connect to arduino
    
    inf = str(x_hori) + ' ' + str(y_hori) + ';'
    arduino.write(bytes(inf, 'utf-8')) 
    time.sleep(2)

    mpc_x = MPC("x", isPath, 8, -0.177, 1, 0.0005,file)
    mpc_y = MPC("y", isPath, 8, 0.14, 1, 0.0005,file)
    while True:
        (isStop, frame, output, x_1 ,y_1, timestamp, homo) = control_queue.get()     
        if isStop:
            inf = str(x_hori) + ' ' + str(y_hori) + ';'
            arduino.write(bytes(inf, 'utf-8'))   
            arduino.close()
            plot_queue.put(params.data())
            break
        
        params.calculate(x0, y0, x_1, y_1, timestamp)
        #print("v",params.vx_now,params.vy_now)
        print("pre",params.x_pre,params.y_pre)
        u_pre, v_pre = labiutils.mappingx2u(homo, [params.x_pre, params.y_pre])
        u_pre = round(u_pre)
        v_pre = round(v_pre)
        frame_queue.put((frame, output, y0, x0, u_pre, v_pre))
        #print("reach", params.is_reach, "still:", params.is_still)
        
        if is_pre:
            time_now = time.time()
            delay_now = time_now - timestamp + (38.05+0)/1000
            a_x = mpc_x.last_u*mpc_x.gravity*mpc_x.K2*5/7
            a_y = mpc_y.last_u*mpc_y.gravity*mpc_y.K2*5/7
            x_pos_now = params.x_c + params.vx_now * delay_now + 0.5*a_x*delay_now*delay_now
            y_pos_now = params.y_c + params.vy_now * delay_now + 0.5*a_y*delay_now*delay_now
            xk = np.array([[x_pos_now], [params.vx_now], [mpc_x.last_u]])
            yk = np.array([[y_pos_now], [params.vy_now], [mpc_y.last_u]])
            
            if params.is_first:
                params.control_time = time_now 
                t0 = 0.0
            else:
                t0 = time_now - params.control_time
        else:
            xk = np.array([[params.x_c], [params.vx_now], [mpc_x.last_u]])
            yk = np.array([[params.y_c], [params.vy_now], [mpc_y.last_u]])
            
            if params.is_first:
                params.control_time = timestamp 
                t0 = 0.0
            else:
                t0 = timestamp - params.control_time 
        tilt_x = mpc_x.MPCcalRealtime(xk, t0)
        tilt_y = mpc_y.MPCcalRealtime(yk, t0)
 
        tilt_x = tilt_x * 180 / math.pi
        tilt_y = tilt_y * 180 / math.pi
        print('\033[31m' + str(round(tilt_x,2)) +" "+ str(round(tilt_y,2)) + '\033[0m')
        tilt_x += x_hori
        tilt_y += y_hori
        print("before com tilt_x:", round(tilt_x,2), "tilt_y:", round(tilt_y,2))
       
        params.update()
        
        tilt_x = round(tilt_x)
        tilt_y = round(tilt_y)
        inf = str(tilt_x) + ' ' + str(tilt_y) + ';'
        arduino.write(bytes(inf, 'utf-8'))
        cts.append(time.time())
        tx.append(tilt_x)   
        ty.append(tilt_y)
#%% 
def img_process():
    total_time = []
    cnn_time = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print("java connected") 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"Running on your {gpu_name} (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on your CPU")
    
    model = CNN(3 if three_images else 1).to(device)
    
    if three_images:
        print("Running CueNet V2")
        model.load_state_dict(torch.load('big_ds_3_images.pt', map_location=device)['state_dict'])
    else:
        print("Running CueNet V1")
        model.load_state_dict(torch.load('big_ds_1_image.pt', map_location=device)['state_dict'])
    print("network loaded") 
    
    print("init homo")
    corners.clear()
    while True:
        receive_data = sock.recv(buf_size)
        jobj = javaobj.loads(receive_data)
        array = np.asarray(jobj.values)
        frame_number = jobj.counter
        frame = transform(array, jobj.channel)
        frame = undistort(frame).reshape(height, width, 1)
        frame_queue.put((frame, 0, -1, -1, -1, -1))  
        if len(corners) == 4:
            break
    print("init homo done")
     
    homo = labiutils.calHomou2x(corners)
    
    controlThread = threading.Thread(target = control)
    controlThread.start()
    print("control thread started")  
    
    warmimg = torch.load("warm.pt")

    for i in range(20):
        output1 = model(warmimg.to(torch.float32).to(device))
        output1 = output1.cpu().detach().numpy() * 70     
        output1 = np.transpose(output1, (1, 2, 0)) 
    for i in range(20):
        receive_data1 = sock.recv(buf_size)
        jobj1 = javaobj.loads(receive_data1)
        array = np.asarray(jobj1.values)
         
        frame_w = transform(array)
        frame_w = undistort(frame_w).reshape(height, width, 1)
    print("warm-up finished")
    
    print("tracking starts running")
    last_frame_number = 0
    frames = []
    while True:
        if stop_queue.empty():
            with Timer('overall consumer loop'):
                start_time = time.time()
                with Timer('deserialize and reshape'):
                    receive_data = sock.recv(buf_size)
                    jobj = javaobj.loads(receive_data)
                    timestamp = jobj.timestampMsEpoch
                    
                    array = np.asarray(jobj.values)
                    
                    frame_number = jobj.counter
                    
                    frame = transform(array, jobj.channel)
                    frame = undistort(frame).reshape(height, width, 1)

                    dropped_frames = frame_number - last_frame_number - 1
                    if dropped_frames > 0:
                        print('Dropped', str(dropped_frames),'frames from producer')
                    last_frame_number = frame_number
                    #print("frame number",frame_number)
                    
                    if three_images:
                        frames.insert(0, frame)
                        if len(frames) < 3:
                            continue
                        img = torch.from_numpy(np.asarray(frames).astype(np.float64)/255.0)
                        img = img.reshape(1, 3, height, width)
                        frames.pop()
                    else:
                        img = frame.astype(np.float64) / 255.0
                        img = torch.from_numpy(img)
                        img = img.reshape(1, 1, height, width)    
   
                cnn_start = time.time()  
                output = model(img.to(torch.float32).to(device))
      
                output = output.cpu().detach().numpy() * 70     
                output = np.transpose(output, (1, 2, 0)) 
                cnn_end = time.time() 
                cnn_time.append(cnn_end - cnn_start)
                
                u_n, v_n = labiutils.findCenter(output)
                x_n, y_n = labiutils.mappingu2x(homo, [u_n,v_n])
                control_queue.put((False, frame, output, x_n, y_n, timestamp/1000, homo))
                print("xn",x_n,y_n)
                xs.append(x_n)
                ys.append(y_n)
                #delay = int(round(time.time() * 1000)) - timestamp
                #print("delay", delay)
                #with Timer('producer->consumer inference delay', delay = delay, show_hist = True):
                 #   pass

                total_time.append(time.time() - start_time)
        else:     
            control_queue.put((True, None, None, None, None, None, None))
            runtime_queue.put((np.mean(total_time),np.mean(cnn_time)))
            sock.close()
            break
#%%
tx = []
ty = []
cts = []
xs = []
ys = []
corners = []

try:
    net_name = "CueNet V2" if three_images else "CueNet V1"
    cv2.namedWindow(net_name, cv2.WINDOW_NORMAL)   
    cv2.resizeWindow(net_name, 2 * width , 2 * height) 
    cv2.setMouseCallback(net_name, mouse_callback)
    
    stop_queue = queue.Queue()
    control_queue = queue.Queue()
    frame_queue = queue.Queue()
    plot_queue = queue.Queue()
    runtime_queue = queue.Queue()
    
    processThread = threading.Thread(target = img_process)
    processThread.start()
    time.sleep(0.2)
    print("show thread started")  
    
    while True:
        (frame, output, y_g, x_g, x_pre, y_pre) = frame_queue.get()
        b = frame.astype(np.float64)
        g = frame.astype(np.float64)
        r = frame.astype(np.float64) + output * 255    
        if x_g > -1 and x_pre > -1:
            b[y_g, x_g, 0] = 255
            g[int(round(y_pre)), int(round(x_pre)), 0] = 255
        out_img = cv2.merge([b,g,r]) / 255.0                          
        cv2.imshow(net_name, out_img)
        cv2.waitKey(1)
    
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    stop_queue.put("Stop!")   
    (total, cnn) = runtime_queue.get()
    (accuracy, t_0, t_list, x_error, y_error) = plot_queue.get()
    mpc_x = MPC("x", isPath, 5, -0.177,1, 0.0001,file)
    mpc_y = MPC("y", isPath, 6, 0.14,1, 0.0001,file)
    print('\033[31m' + "Press Ctrl-C to terminate while statement" + '\033[0m')
    print("total time", total, "\n", "cnn time", cnn, "\n", "accuracy", accuracy)
    fig, axs = plt.subplots(2)
    fig.suptitle('MPC Curves of two axes')
    xer = [ -x for x in x_error]
    yer = [ -y for y in y_error]
    xre = []
    yre = []
    for i in t_list:
        xre.append(mpc_x.t2xy(i))
        yre.append(mpc_y.t2xy(i))
    axs[0].plot(t_list, xer)
    axs[0].plot(t_list, xre)
    axs[1].plot(t_list, yer)
    axs[1].plot(t_list, yre)
            
    t_list = [i + t_0 for i in t_list]
    np.save("events/data/mpc/t_list.npy", t_list)
    np.save("events/data/mpc/cts.npy", cts)
    np.save("events/data/mpc/xs.npy", xs)
    np.save("events/data/mpc/ys.npy", ys)
    np.save("events/data/mpc/tx.npy", tx)
    np.save("events/data/mpc/ty.npy", ty)
