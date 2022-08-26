import numpy as np
import cv2
import math
import scipy.interpolate
#%%
def clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif (upper is not None) and (value > upper):
        return upper
    elif (lower is not None) and (value < lower):
        return lower
    return value
#%%
def saveRGBfromGray(frame, name, x = None, y = None):
    b = frame.astype(np.float64)
    g = frame.astype(np.float64)
    r = frame.astype(np.float64)
    if x is not None and y is not None:
        b[y, x, 0] = 255     
    out_img = cv2.merge([b,g,r])   
    cv2.imwrite(name, out_img)
#%%
def round2(v):
    low = int(v*10000)
    dd = low % 10000
    re = (low - low%10000)/10000
    if  dd < 2500:
        return re
    elif dd >= 2500 and dd < 7500:
        return re + 0.5
    else:
        return re + 1.0
#%%
def round4(v):
    low = int(v*10000)
    dd = low % 10000
    re = (low - low%10000)/10000
    if  dd < 1250:
        return re
    elif dd >= 1250 and dd < 3750:
        return re + 0.25
    elif dd >= 3750 and dd < 6250:
        return re + 0.5
    elif dd >= 6250 and dd < 8750:
        return re + 0.75
    else:
        return re + 1.0
#%%
def round8(v):
    low = int(v*10000)
    dd = low % 10000
    re = (low - low%10000)/10000
    if dd < 625:
        return re
    elif dd >= 625 and dd < 1875:
        return re + 0.125
    elif dd >= 1875 and dd < 3125:
        return re + 0.25
    elif dd >= 3125 and dd < 4375:
        return re + 0.375
    elif dd >= 4375 and dd < 5625:
        return re + 0.5
    elif dd >= 5625 and dd < 6875:
        return re + 0.625
    elif dd >= 6875 and dd < 8125:
        return re + 0.75
    elif dd >= 8125 and dd < 9375:
        return re + 0.875
    else:
        return re + 1.0
#%%
def findCorners(img, ranges):
    results = []
    for cor in range(len(ranges)):
        x1, y1, x2, y2 = ranges[cor]
        xd = 0
        yd = 0
        isstop = False
        for i in range(x2-x1):
            if isstop:
                break
            for j in range(y2-y1):
                x = x1 + i
                y = y1 + j
                low = True
                for r in [-1, 0, 1]:
                    for c in [-1, 0, 1]:
                        if r == 0 and c == 0:
                            continue
                        low = low & (img[y + r][x + c] - img[y][x] > 20)
                if low :
                    xd = x
                    yd = y
                    isstop = True
                    break
        results.append([xd,yd])
    return results
#%%
def calHomou2x(ps):
    assert len(ps) == 4
    A = np.empty((0,8), float);
    corners = np.array([[0,0],[0.29,0],[0.29,0.245],[0,0.245]])
    b = corners.reshape(8,)
    for i in range(4):
        u = ps[i][0]
        v = ps[i][1]
        x = corners[i][0]
        y = corners[i][1]
        r1 = np.array([[u,v,1,0,0,0,-u*x,-v*x]])
        r2 = np.array([[0,0,0,u,v,1,-u*y,-v*y]])
        A = np.append(A, r1, axis=0)
        A = np.append(A, r2, axis=0)
    hs = np.linalg.solve(A, b)
    homo = np.array([[hs[0],hs[1],hs[2]],[hs[3],hs[4],hs[5]],[hs[6],hs[7],1]])
    return homo
    
def mappingu2x(homo, uv):
    vec = np.array([[uv[0]], [uv[1]], [1]])
    xy = np.matmul(homo, vec)
    result = [xy[0][0]/xy[2][0],xy[1][0]/xy[2][0]]
    return result

def mappingx2u(homo, xy):
    x = xy[0]
    y = xy[1]
    vec = np.array([[homo[0][2]-x], [homo[1][2]-y]])
    m = np.array([[homo[2][0]*x-homo[0][0],homo[2][1]*x-homo[0][1]],[homo[2][0]*y-homo[1][0],homo[2][1]*y-homo[1][1]]])
    rm = np.linalg.inv(m)
    uv = np.matmul(rm, vec)
    return [uv[0][0],uv[1][0]]
    '''
    vec = np.array([[xy[0]], [xy[1]], [1]])
    m = np.linalg.inv(homo)
    uv = np.matmul(m, vec)
    result = [uv[0][0]/uv[2][0],uv[1][0]/uv[2][0]]
    return result
    '''
#%%
def findCenter(output, is_weighted = True):
    center_output = np.where(output == np.amax(output)) 
    x_t = center_output[1][0]
    y_t = center_output[0][0]
    xsum = 0.0
    ysum = 0.0
    msum = 0.0
    for i in range(15):
        for j in range(17):
            x = i - 8 + x_t
            y = j - 8 + y_t
            xsum += output[y][x][0] * x
            ysum += output[y][x][0] * y
            msum += output[y][x][0]
    xr = round8(xsum/msum)
    yr = round8(ysum/msum)
    v = output[y_t][x_t][0]
    if is_weighted:
        return xr, yr
    mind = 1.0
    xd = 0
    yd = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i ==0 and j == 0:
                continue
            if v - output[y_t+j][x_t+i][0] < mind:
                mind = v - output[y_t+j][x_t+i][0]
                xd = i
                yd = j
    if mind < 0.008:
        x_t = x_t + 0.5 * xd
        y_t = y_t + 0.5 * yd
    return x_t, y_t    
#%%
def findCircle(image, xp, yp):
    r_x = 10
    r_y = 10
    xmin = round(xp - r_x)
    xmax = round(xp + r_x)
    ymin = round(yp - r_y)
    ymax = round(yp + r_y)
    img1 = image[ymin:ymax, xmin:xmax]
    img1 = cv2.GaussianBlur(img1,(3,3),cv2.BORDER_DEFAULT)
    img1 = cv2.medianBlur(img1, 3)
    th = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = [] 
    for i, c in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = [x + xmin, y + ymin]    
        if radius > 4 and radius < 7:
            centers.append(center)
    if len(centers) == 1:
        return round4(centers[0][0]),round4(centers[0][1])
    else:
        return xp, yp
#%%
def genMap(path, v = 0.02):
    x = []
    y = []
    tlim = [0]
    t = 0.0
    for p in path:
        x.append(p[0])
        y.append(p[1])
    for i in range(len(x)-1):
        t = t + math.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2)/v;
        tlim.append(t)
    tend = round(1000*tlim[-1])/1000
    tre = np.arange(0,tend,0.001)
    fx = scipy.interpolate.interp1d(tlim,x,kind ="linear")   
    fy = scipy.interpolate.interp1d(tlim,y,kind ="linear")
    xre = fx(tre)
    yre = fy(tre)
    return tre, xre, yre
    
def meanerror(x,y,r1,r2):
    x=x[r1:r2]
    y=y[r1:r2]
    e = []
    for i in range(len(x)):
        e.append(abs(x[i]-y[i]))
    return np.mean(e)