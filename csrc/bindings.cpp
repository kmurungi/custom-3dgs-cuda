#include <torch/extension.h>
#include "projection.h"
#include "rasterize.h"
#include "SH.h"
#include "projection_backward.h"
#include "rasterize_backward.h"
#include "SH_backward.h"


//projection function
torch::Tensor project(
    torch::Tensor means3D, 
    torch::Tensor proj_matrix) 
{
    // 1. Safety Checks (Must be CUDA tensors)
    TORCH_CHECK(means3D.is_cuda(), "means3D must be a CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be a CUDA tensor")

    // 2. Call the actual CUDA execution function
    return project_gaussians_to_2d(means3D, proj_matrix);
}

//spherical harmonics function 


//rasterize function


//project_backward

//spherical harmonics backward 

//rasterize_backward 



//PyBind11 Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project", &project, "3DGS Projection (CUDA)");
    m.def("SH", &spherical_harmonics,"3DGS Spherical Harmonics (CUDA)");
    m.def("rasterize", &rasterize,"3DGS Spherical Harmonics (CUDA)");

    m.def("project_backward", &project_backward);
    m.def("SH_backward", &spherical_harmonics_backward);
    m.def("rasterize_backward", &rasterize_backward);

}