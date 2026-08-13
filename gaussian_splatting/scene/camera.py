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
        Samples Camera view for forward pass.

        Returns:
            list of dictionaries with pose / intrinsics (and image_path if set)
        """
        image_paths = getattr(self, "image_paths", [None] * self.N)
        image_names = getattr(self, "image_names", [None] * self.N)

        camera_views = []
        for i in range(self.N):
            camera_views.append(
                {
                    "R": self.R[i],
                    "T": self.t[i],
                    "fx": self.fx[i],
                    "fy": self.fy[i],
                    "cx": self.cx[i],
                    "cy": self.cy[i],
                    "image_path": image_paths[i],
                    "image_name": image_names[i],
                }
            )
        return camera_views
