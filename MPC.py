import numpy as np
import random
import math
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import time
import labiutils

class MPC():
    def __init__(self, name, isPath, umax, scale, q, r, path = "path.npy"):
        self.N = 100
        self.tt= 0.005
        self.tn = round(0.05/self.tt) - 1
        self.K1 = 180
        self.K2 = scale
        self.gravity = 9.805
        self.ny = 1
        self.nu = 1
        self.nx = 3
        self.umax_r = umax
        self.dumax_r = 3
        self.umax = math.pi *umax/180
        self.dumax = math.pi *3/180
        self.x0 = np.array([[0], [0], [0]])
        self.u = u = np.ones(self.N).reshape(self.N,1) * (-math.pi *15/180)
        self.A = np.array([[1, self.tt, 0], [0, 1, (5/7)*self.tt*self.gravity*self.K2], [0, 0, 1-self.K1*self.tt]])
        self.B = np.array([[0], [0], [self.K1*self.tt]])
        self.C = np.array([[1, 0, 0]])
        self.Q = np.array([q])
        self.R = np.array([r])
        self.w = np.ones(self.N + 1).reshape(self.N + 1,1) * 0.05
        self.H = None
        self.g = None
        self.V = None
        self.S = None
        self.lb = None
        self.ub = None
        self.AM = None
        self.bM = None
        self.GM = None
        self.hM = None
        self.isd = True
        self.isPath = isPath
        self.name = name
        self.last_u = 0.0
        self.file = path
        self.MatrixCal()
        self.constraints()
        self.genpath(path)
        
    def MatrixCal(self):
        N = self.N
        Q = np.kron(np.eye(N + 1,dtype = float), self.Q)
        R = np.kron(np.eye(N, dtype = float), self.R)
    
        temp = self.C
        V = temp
        S0 = np.zeros((self.ny,self.nu))
    
        for i in range(N):
            M = np.matmul(temp, self.B)
            S0 = np.append(S0, M, axis=0)
            temp = np.matmul(temp, self.A)
            V = np.append(V, temp, axis=0)
        
        temp = S0
        S = temp
        for i in range(N-1):
            temp = np.append([[0]], temp[:-1], axis = 0)
            S = np.append(S, temp, axis = 1)
            
        ST = np.transpose(S)
        Htemp = np.matmul(ST, Q)
        H = np.matmul(Htemp, S) + R
        g = np.matmul(Q, S)
        g = np.transpose(g)
        
        self.H = H
        self.g = g
        self.V = V
        self.S = S
        
    def constraints(self):
        N = self.N
        L = np.zeros(N).reshape(N,)
        self.lb = L - self.umax
        self.ub = L + self.umax
    
        self.bM = np.zeros(N).reshape(N,)
        self.hM = np.ones(2*N-2).reshape(2*N-2,) * self.dumax
        AM = np.zeros((N,N))
        GM = np.zeros((2*N-2,N))
        
        for i in range(N):
            k = int(i/self.tn)
            if i + k + 1 > N-1:
                break
            AM[i][i + k] = 1
            AM[i][i + k + 1] = -1
            
        for i in range(N-1):
            GM[2*i][i] = 1
            GM[2*i][i + 1] = -1
            GM[2*i + 1][i] = -1
            GM[2*i + 1][i + 1] = 1
            
        self.AM = AM
        self.GM = GM
    
    def MPCcal(self, xk, w):
        y = np.matmul(self.V, xk) - w
        f = np.matmul(self.g, y)
        #start = time.time()
        if self.isd:
            uu = solve_qp(P = self.H, q = f, G = self.GM, h = self.hM, A = self.AM, b = self.bM, lb = self.lb, ub = self.ub, solver="osqp")
        else:
            uu = solve_qp(P = self.H, q = f, A = self.AM, b = self.bM, lb = self.lb, ub = self.ub, solver="osqp") 
        #print(time.time() - start)
        #print("QP solution: x = {}".format(uu))
        return uu
    
    def MPCcalRealtime(self, xk, tp):
        w = np.array([self.t2xy(tp + self.tt * i) for i in range(self.N + 1)]).reshape(self.N+1,1)
        uu = self.MPCcal(xk, w)
        self.last_u = uu[0]
        return uu[0]
        
    def ploty(self, x, u):
        assert x.shape == (self.nx, 1)
        assert u.shape == (self.N * self.nu, 1)
        y = np.matmul(self.V, x) + np.matmul(self.S, u)
        t = [i*0.01 for i in range(self.N+1)] 
        plt.plot(t,y)
        return y
    
    def xrand(self, x_now):
        randx = (random.random()-0.5)*2*0.05 + 1
        randv = (random.random()-0.5)*2*0.1 + 1
        randxx = (random.random()-0.5)*2*0.0004
        randvv = (random.random()-0.5)*2*0.007
        x_now[0][0] = x_now[0][0] * randx + randxx
        x_now[1][0] = x_now[1][0] * randv + randvv
        return x_now
    
    def t2y(self, t):
        y1 = 0.0825
        y2 = 0.1625
        td = 4
        t = round(1000*(t % (4*td)))
        if 0 <= t and t < td*1000:
            y = y1
        elif td*1000 <= t and t < td*2000:
            y = y1 + (t-td*1000)*(y2-y1)/(td*1000)
        elif td*2000 <= t and t < td*3000:
            y = y2  
        else:
            y = y2 - (t-td*3000)*(y2-y1)/(td*1000)
        return y
    
    def t2x(self, t):
        x1 = 0.095
        x2 = 0.195
        td = 4
        t = round(1000*(t % (4*td)))
        if 0 <= t and t < td*1000:
            x = x1 + t*(x2-x1)/(td*1000)
        elif td*1000 <= t and t < td*2000:
            x = x2
        elif td*2000 <= t and t < td*3000:
            x = x2 - (t-td*2000)*(x2-x1)/(td*1000)
        else:
            x = x1
        return x
    
    def genpath(self, path):
        self.p = np.load(path)
        self.tre, self.xre, self.yre = labiutils.genMap(self.p)
        
    def t2xy(self, t):
        if self.isPath:      
            t = round(t,3)
            if t > self.tre[-1]:
                t = self.tre[-1]
            i = round(t * 1000)
            if self.name == "x":
                return self.xre[i]
            elif self.name == "y":
                return self.yre[i]
        else:
            if self.name == "x":
                return self.t2x(t)
            elif self.name == "y":
                return self.t2y(t)