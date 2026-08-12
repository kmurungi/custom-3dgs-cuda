#include <torch/extension.h>
#include "projection.h"
#include "rasterize.h"
#include "SH.h"
#include "projection_backward.h"
#include "rasterize_backward.h"
#include "SH_backward.h"


//projection function
std::tuple<torch::Tensor, torch::Tensor> project(
    int total_gaussians, 
    const torch::Tensor  mu3d,               // N x 3
    const torch::Tensor  q3d,                // N x 4 (w, x, y, z)
    const torch::Tensor  s3d,                // N x 3
    const torch::Tensor  camera_rotation,   // 3 x 3 matrix rows
    const torch::Tensor  camera_translation,// 3 x 1 vector
    float fx, float fy, float cx, float cy
){
   
    // Call the actual CUDA execution function
    return project_gaussians_to_2d(
        total_gaussians, 
        mu3d,               // N x 3
        q3d,                // N x 4 (w, x, y, z)
        s3d,                // N x 3
        camera_rotation,   // 3 x 3 matrix rows
        camera_translation,// 3 x 1 vector
        fx, fy, cx, cy);
}

//spherical harmonics function 
int spherical_harmonics( 

){ 

    return launch_spherical_harmonics_kernel(); 
}

//rasterize function

//project_backward

//spherical harmonics backward 

//rasterize_backward 



//PyBind11 Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project", &project, "3DGS Projection (CUDA)");
    m.def("SH", &spherical_harmonics,"3DGS Spherical Harmonics (CUDA)");
    // m.def("rasterize", &rasterize,"3DGS Spherical Harmonics (CUDA)");

    // m.def("project_backward", &project_backward);
    // m.def("SH_backward", &spherical_harmonics_backward);
    // m.def("rasterize_backward", &rasterize_backward);

}