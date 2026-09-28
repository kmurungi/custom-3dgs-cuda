import torch as t

class camera: 

    # for N images
    def __init__(self, N): 
        self.N = N

        # extrinsics
        self.R = t.zeros(N, 3, 3, dtype=t.float32)  # world-to-camera rotation
        self.t = t.zeros(N, 3, dtype=t.float32)  # world-to-camera translation

        # intrinsics
        self.fx = t.zeros(N, 1, dtype=t.float32)  # focal length x
        self.fy = t.zeros(N, 1, dtype=t.float32)  # focal length y
        self.cx = t.zeros(N, 1, dtype=t.float32)  # principal point x
        self.cy = t.zeros(N, 1, dtype=t.float32)  # principal point y

    def sample_camera_view(self):
        """Sample one view per camera, including image_path when it is set."""
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
