import torch as t

class camera: 
    def __init__(self, N): 
        # extrinsics 
        self.R = t.zeros(N, 3, 3) # rotation from world to camera
        self.t = t.zeros(N, 3) # translation from world to camera
        
        # intrinsics
        self.fx = t.zeros(N, 1) # x focal length
        self.fy = t.zeros(N, 1) # y focal length
        self.cx = t.zeros(N, 1) # x principle component
        self.cy = t.zeros(N, 1) # y principle component 
