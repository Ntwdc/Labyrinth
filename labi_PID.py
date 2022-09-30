import javaobj
import serial
import socket
import cv2 
import time
import torch
import queue 
import threading
import numpy as np
import matplotlib.pyplot as plt
from CNN_ER_FA import CNN
from PID import PID
from globals_and_utils import Timer
from numba import jit
import tkinter as tk
from params import Runparams
import random
import labiutils
#%%
x_direction = -1
y_direction = 1
width = 240
height = 180
port = 7777
buf_size = 63000
host = 'localhost'
three_images = True
is_auto = False
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
        target_queue.put((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  
        corners.append([x,y])
        print(x,y)
    if event == cv2.EVENT_MBUTTONDOWN:
        path.append([x, y])
        print(x,y)
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
    x_hori = 86
    y_hori = 90
    tilt_x = x_hori
    tilt_y = y_hori
    seq = 0
    count = 0
    accumulationx = 0.0
    accumulationy = 0.0
    is_apped = False
    is_setnext = True
    is_pre = False
    params = Runparams(0.29, 0.245)
    
    print("control thread starts")
    arduino = serial.Serial('COM3', 115200, timeout = 5)
    print("arduino connected") # connect to arduino
    
    inf = str(x_hori) + ' ' + str(y_hori) + ';'
    arduino.write(bytes(inf, 'utf-8')) 
                
    limits_x = (-15, 15)
    limits_y = (-15, 15)
    pid_x = PID(sample_time = None, output_limits = limits_x, setpoint = -1)
    pid_y = PID(sample_time = None, output_limits = limits_y, setpoint = -1) 
    print("PID set")
    rat = 1.26
    p1 = 70.0
    i1 = 20.0
    d1 = 70.0
    pid_x.tunings = (x_direction * p1, x_direction * i1, x_direction * d1)
    pid_y.tunings = (y_direction * p1*rat, y_direction * i1*rat, y_direction * d1*rat)   
    print("PID gains X:" ," p:", pid_x.Kp, " i:", pid_x.Ki, " d:", pid_x.Kd)
    print("PID gains Y:" ," p:", pid_y.Kp, " i:", pid_y.Ki, " d:", pid_y.Kd)

    while True:
        (isStop, frame, output, x_1 ,y_1, timestamp, homo) = control_queue.get()
        if isStop:
            inf = str(x_hori) + ' ' + str(y_hori) + ';'
            arduino.write(bytes(inf, 'utf-8'))   
            arduino.close()
            plot_queue.put(params.data())
            break
        #set next point
        if not is_auto:
            if not target_queue.empty():
                u0, v0 = target_queue.get()
                x0, y0 = labiutils.mappingu2x(homo, [u0, v0]) 
                print("next point", "u", u0, "v", v0,"x:", x0, "y", y0)
                if not target_queue.empty():
                    target_queue.queue.clear()
        else:
            if is_setnext and seq < len(path):
                x0, y0 = path[seq]
                u0, v0 = labiutils.mappingx2u(homo, [x0, y0]) 
                u0 = round(u0)
                v0 = round(v0)
                seq += 1
                is_setnext = False
                last_set = time.time()
                print("next point", "u", u0, "v", v0,"x:", x0, "y", y0)
           
        params.calculate(x0, y0, x_1, y_1, timestamp)
        u_pre, v_pre = labiutils.mappingx2u(homo, [params.x_pre, params.y_pre])
        u_pre = round(u_pre)
        v_pre = round(v_pre)
        print("frame",round(time.time(),3))
        frame_queue.put((frame, output, v0, u0, u_pre, v_pre))
        
        if params.is_next:
            pid_x.setpoint = x0
            pid_y.setpoint = y0
            is_apped = False
        
        print("reach", params.is_reach, "still:", params.is_still)
        
        ms = 0
        if is_auto and (time.time()-last_set) > 1.5:
            is_setnext = True
        if params.is_reach:
            count += 1
            if not is_apped and count > ms:
                inc.append([x0,y0,x_1,y_1,tilt_x,tilt_y])
                is_apped = True
                #is_setnext = True
                count = 0
            params.update()
            params.control_time = time.time()
            
            tx.append(tilt_x)   
            ty.append(tilt_y)
            px.append(pid_x._proportional)
            py.append(pid_y._proportional)
            ix.append(pid_x._integral)
            iy.append(pid_y._integral)
            dx.append(0.0)
            dy.append(0.0)
            continue
        else:
            count = 0
            if is_pre:    
                time_now = time.time()
                delay_now = time_now - timestamp + 38.05/1000
                x_pos_now = params.x_c + params.vx_now * delay_now 
                y_pos_now = params.y_c + params.vy_now * delay_now 
                td_pre = params.t_d if params.is_first else (time_now - params.control_time)
                params.control_time = time_now
            else:
                x_pos_now = params.x_c 
                y_pos_now = params.y_c 
                td_pre = params.t_d 
                
            tilt_x = pid_x(x_pos_now, dt = td_pre) 
            tilt_y = pid_y(y_pos_now, dt = td_pre)
            print("before com tilt_x:", round(tilt_x,2), "tilt_y:", round(tilt_y,2))
            
            co = 0.4
            if params.is_still:
                if params.is_still_last:
                    compx, compy = comp(params.delta_x, params.delta_y)
                    accumulationy += 2 * co * compy 
                    accumulationx += 2 * co * compx
            else:
                if not params.is_still_x:
                    accumulationx = 0.0                   
                else:
                    if params.is_still_x_last:
                        accumulationx = accumulationx + co
                if not params.is_still_y:
                    accumulationy = 0.0                   
                else:
                    if params.is_still_y_last:
                        accumulationy = accumulationy + co
                    
            compensationx = x_direction * (0.0 + accumulationx)
            compensationy = y_direction * (0.0 + accumulationy)
            print("errorx",params.x0-params.x1)
            print("errory",params.y0-params.y1)
            print("pid x:", round(pid_x._proportional,4), round(pid_x._integral,4), round(pid_x._derivative,4), compensationx)
            print("pid y:", round(pid_y._proportional,4), round(pid_y._integral,4), round(pid_y._derivative,4), compensationy)
            px.append(pid_x._proportional)
            py.append(pid_y._proportional)
            ix.append(pid_x._integral)
            iy.append(pid_y._integral)
            dx.append(pid_x._derivative)
            dy.append(pid_y._derivative)
            
            
            if params.is_still_x and abs(params.x1 - params.x0) > 0:
                tilt_x = tilt_x + compensationx if params.x1 < params.x0 else tilt_x - compensationx
            if params.is_still_y and abs(params.y1 - params.y0) > 0:
                tilt_y = tilt_y + compensationy if params.y1 < params.y0 else tilt_y - compensationy  
            print("after com tilt_x:", round(tilt_x,2), "tilt_y:", round(tilt_y,2))         
            params.update()
                
        tilt_x = round(tilt_x)
        tilt_y = round(tilt_y)
        tilt_x = labiutils.clamp(tilt_x, limits_x)
        tilt_y = labiutils.clamp(tilt_y, limits_y)
        tilt_x += x_hori
        tilt_y += y_hori
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
    print("init target")
    target_queue.queue.clear()
    while True:
        receive_data = sock.recv(buf_size)
        jobj = javaobj.loads(receive_data)
        array = np.asarray(jobj.values)
        frame_number = jobj.counter
        frame = transform(array, jobj.channel)
        frame = undistort(frame).reshape(height, width, 1)
        frame_queue.put((frame, 0, -1, -1, -1, -1))  
        if not target_queue.empty():
            break
    print("init target done")      
    if is_auto and len(path) == 0:
        print("No path")
        return
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
                
                #u_n, v_n = labiutils.findCenter(output)
                center_output = np.where(output == np.amax(output)) 
                u_n = center_output[1][0]
                v_n = center_output[0][0]
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
px = []
py = []
tx = []
ty = []
ix = []
iy = []
dx = []
dy = []
inc = []
path = []
corners = []
cts = []
xs = []
ys = []
pa = []
'''
for i in range(19):
    for j in range(13):
        x = 0.06 + 0.01*i
        y = 0.06 + 0.01*j
        pa.append((x,y))

for i in range(10):
    path = path + pa
random.shuffle(path)
'''
try:
    net_name = "CueNet V2" if three_images else "CueNet V1"
    cv2.namedWindow(net_name, cv2.WINDOW_NORMAL)   
    cv2.resizeWindow(net_name, 2 * width , 2 * height) 
    cv2.setMouseCallback(net_name, mouse_callback)
    
    target_queue = queue.Queue()
    stop_queue = queue.Queue()
    control_queue = queue.Queue()
    frame_queue = queue.Queue()
    plot_queue = queue.Queue()
    runtime_queue = queue.Queue()
    cali_queue = queue.Queue()
    
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
            b[round(y_g), round(x_g), 0] = 255
            g[round(y_pre), round(x_pre), 0] = 255
        out_img = cv2.merge([b,g,r]) / 255.0                          
        cv2.imshow(net_name, out_img)
        cv2.waitKey(1)
    
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    stop_queue.put("Stop!")   
    (total, cnn) = runtime_queue.get()
    (accuracy, t_0, t_list, x_error, y_error) = plot_queue.get()
    print('\033[31m' + "Press Ctrl-C to terminate while statement" + '\033[0m')
    print("total time", total, "\n", "cnn time", cnn, "\n", "accuracy", accuracy)
    fig, axs = plt.subplots(2)
    fig.suptitle('PID Transient Curves of two axes')
    axs[0].plot(t_list, x_error)
    axs[1].plot(t_list, y_error)
