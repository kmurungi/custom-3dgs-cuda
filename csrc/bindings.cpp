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
torch::Tensor spherical_harmonics( 
    int total_gaussians, 
    const torch::Tensor camera_position, // 3 x 1
    const torch::Tensor mu_world, // N x 3 
    const torch::Tensor albedo_coeff, // N x 3 
    const torch::Tensor illumination_coeff // (N, 15, 3) flattened to 1 contiguous array 
){

    return launch_spherical_harmonics_kernel(
        total_gaussians,
        camera_position, 
        mu_world,
        albedo_coeff,
        illumination_coeff
    ); 
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