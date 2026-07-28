import torch as t

class camera: 

    # for N images
    def __init__(self, N): 
        self.N = N

        # extrinsics 
        self.R = t.zeros(N, 3, 3) # rotation from world to camera
        self.t = t.zeros(N, 3) # translation from world to camera
        
        # intrinsics
        self.fx = t.zeros(N, 1) # x focal length
        self.fy = t.zeros(N, 1) # y focal length
        self.cx = t.zeros(N, 1) # x principle component
        self.cy = t.zeros(N, 1) # y principle component 

    def sample_camera_view(self): 
        """
        Samples Camera view for forward pass 
        
        Returns: 
            list of dictionary with tensor values for each img
        """
        camera_views = [ {
            "R" : self.R[i], 
            "T" : self.t[i], 
            "fx" : self.fx[i], 
            "fy" : self.fy[i], 
            "cx" : self.cx[i],
            "cy" : self.cy[i]
        } for i in range(self.N)]

        return camera_views
        