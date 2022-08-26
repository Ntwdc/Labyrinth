def clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif (upper is not None) and (value > upper):
        return upper
    elif (lower is not None) and (value < lower):
        return lower
    return value

class Runparams():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x_last = -1 
        self.y_last = -1
        self.t_last = 0
        self.timestamp = 0.0
        self.x0 = -1
        self.y0 = -1
        self.x1 = 0.0
        self.y1 = 0.0
        self.vx_now = 0.0 
        self.vy_now = 0.0
        self.vx_last = 0.0 
        self.vy_last = 0.0
        self.x_pre = 0.0
        self.y_pre = 0.0
        self.x_c = 0.0
        self.y_c = 0.0
        self.x_d = 0.0
        self.y_d = 0.0
        self.t_d = 0.0
        self.control_time = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.threshold = 15
        self.total_frame = 0
        self.failed_frame = 0
        self.sign_x = 1
        self.sign_y = -1
        self.is_overshoot_x = False
        self.is_overshoot_y = False
        self.is_first = True
        self.is_still = True
        self.is_still_x = True
        self.is_still_y = True
        self.is_still_x_last =True
        self.is_still_y_last =True
        self.is_still_last =True
        self.is_lost = False
        self.is_reach = False
        self.is_next = False
        self.t_list = []
        self.x_error = []
        self.y_error = []

    def isLost(self, num, num1, num2):
        (low, high) = (num1, num2) if num1 < num2 else (num2, num1)
        return num < low - self.threshold or num > high + self.threshold 
        
    def calculate(self, x0, y0, x1, y1, timestamp):
        self.is_next = (self.x0 != x0 or self.y0 != y0)
        
        if self.is_next:
            self.sign_x = 1 if x0 > x1 else -1
            self.sign_y = 1 if y0 > y1 else -1
        
        signx = 1 if x0 > x1 else -1
        signy = 1 if y0 > y1 else -1
        
        self.is_overshoot_x = (self.sign_x + signx == 0)
        self.is_overshoot_y = (self.sign_y + signy == 0)
        
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.timestamp = timestamp
        
        self.delta_x = abs(self.x0 - self.x1)
        self.delta_y = abs(self.y0 - self.y1)
        
        if self.is_first:
            self.x_pre = x1
            self.y_pre = y1
            self.x_c = x1
            self.y_c = y1
            self.t_d = 1e-16
        else:
            '''
            if self.is_reach:
                self.is_still = abs(self.x1 - self.x_last) < 2 and abs(self.y1 - self.y_last) < 2
            else:
            '''
            self.is_still_x = self.x1 == self.x_last
            self.is_still_y = self.y1 == self.y_last
            self.is_still =  self.is_still_x and self.is_still_y
            
            ran = 0.0125 * self.width
            if self.is_reach and self.is_still and not self.is_next:
                self.is_reach = True
            elif self.is_still and abs(self.x1 - self.x0) < ran and abs(self.y1 - self.y0) < ran:
                self.is_reach = True
            else:
                self.is_reach = False
                
            self.x_d = self.x1 - self.x_last
            self.y_d = self.y1 - self.y_last
            self.t_d = self.timestamp - self.t_last

            self.x_pre = clamp(self.x_last + self.t_d * self.vx_last, (0, self.width)) 
            self.y_pre = clamp(self.y_last + self.t_d * self.vy_last, (0, self.height))
            
            self.is_lost = self.isLost(self.x1, self.x_pre, self.x_last) | self.isLost(self.y1, self.y_pre, self.y_last)
            
            if self.is_lost:
                 self.x_c = self.x_pre
                 self.y_c = self.y_pre
                 self.vx_now = 1.0 * self.vx_last 
                 self.vy_now = 1.0 * self.vy_last
            else:
                 self.x_c = x1
                 self.y_c = y1
                 self.vx_now = 1.0 * self.x_d / self.t_d   
                 self.vy_now = 1.0 * self.y_d / self.t_d
        self.info()
        
    def update(self):
        self.total_frame = self.total_frame + 1
        
        self.t_list.append(self.timestamp)
        
        self.x_error.append(self.x0 - self.x1)
        self.y_error.append(self.y0 - self.y1)
                
        if self.is_lost:
            self.failed_frame = self.failed_frame + 1
            
        self.t_last = self.timestamp
        self.is_still_last = self.is_still
        self.is_still_x_last = self.is_still_x
        self.is_still_y_last = self.is_still_y
        if self.is_first:
            self.x_last = self.x1
            self.y_last = self.y1
            self.is_first = False
        else:
            if self.is_lost:
                self.x_last = self.x_pre
                self.y_last = self.y_pre
            else:
                self.x_last = self.x1
                self.y_last = self.y1
                self.vx_last = self.vx_now  
                self.vy_last = self.vy_now
                
    def info(self):
        print("x0", self.x0, "x_now", self.x1, "x_last", self.x_last)
        print("y0", self.y0, "y_now", self.y1, "y_last", self.y_last)
        if self.is_lost:
            print("track failed for this frame")
    
    def data(self):
        accuracy = 1.0 - self.failed_frame / self.total_frame
        t_0 = self.t_list[0]
        for i in range(len(self.t_list)):
            self.t_list[i] = (self.t_list[i] - t_0) 
        return (accuracy,t_0, self.t_list, self.x_error, self.y_error)
