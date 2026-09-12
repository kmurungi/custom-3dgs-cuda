#include <torch/extension.h>
#include "forward/projection.h"
#include "forward/rasterize.h"
#include "forward/SH.h"

#include "backward/delight_backward.h"
#include "backward/projection_backward.h"
#include "backward/rasterize_backward.h"


//projection function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project(
    int total_gaussians,
    const torch::Tensor mu3d,
    const torch::Tensor q3d,
    const torch::Tensor s3d,
    const torch::Tensor camera_rotation,
    const torch::Tensor camera_translation,
    float fx, float fy, float cx, float cy
){
    return project_gaussians_to_2d(
        total_gaussians,
        mu3d,
        q3d,
        s3d,
        camera_rotation,
        camera_translation,
        fx, fy, cx, cy);
}

//spherical harmonics function
torch::Tensor spherical_harmonics(
    int total_gaussians,
    const torch::Tensor camera_position,
    const torch::Tensor mu_world,
    const torch::Tensor albedo_coeff,
    const torch::Tensor illumination_coeff
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
// Returns: rendered_img, sorted_ids, tile_ranges, final_T, n_contrib
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize(
    int num_gaussians,
    const torch::Tensor depths,
    const torch::Tensor mean_2d,
    const torch::Tensor cov_2d,
    const torch::Tensor colors,
    const torch::Tensor alpha,
    int height,
    int width
){
    return launch_rasterization(
        num_gaussians,
        depths,
        mean_2d,
        cov_2d,
        colors,
        alpha,
        height,
        width
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backwards_rasterization(
    const torch::Tensor grad_output,
    const torch::Tensor mean_2d,
    const torch::Tensor cov_2d,
    const torch::Tensor colors,
    const torch::Tensor alpha,
    const torch::Tensor sorted_ids,
    const torch::Tensor tile_ranges,
    const torch::Tensor final_T,
    const torch::Tensor n_contrib,
    int num_gaussians,
    int height,
    int width
){
    return launch_backward_rasterization(
        grad_output,
        mean_2d,
        cov_2d,
        colors,
        alpha,
        sorted_ids,
        tile_ranges,
        final_T,
        n_contrib,
        num_gaussians,
        height,
        width
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backwards_SH(
    const torch::Tensor colors_grad,
    const torch::Tensor mu,
    const torch::Tensor albedo,
    const torch::Tensor k_j,
    const torch::Tensor camera_position
){
    return launch_backward_SH(
        colors_grad,
        mu,
        albedo,
        k_j,
        camera_position
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backwards_projection(
    const torch::Tensor mean_2d_grad,
    const torch::Tensor cov_2d_grad,
    const torch::Tensor mu,
    const torch::Tensor q,
    const torch::Tensor s,
    const torch::Tensor camera_rotation,
    const torch::Tensor camera_translation,
    float fx, float fy, float cx, float cy
){
    return launch_backward_projection(
        mean_2d_grad,
        cov_2d_grad,
        mu,
        q,
        s,
        camera_rotation,
        camera_translation,
        fx, fy, cx, cy
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //forward kernels
    m.def("project", &project, "3DGS Projection (CUDA)");
    m.def("SH", &spherical_harmonics, "3DGS Spherical Harmonics (CUDA)");
    m.def("rasterize", &rasterize, "3DGS Tile Rasterization (CUDA)");

    //backwards kernels
    m.def("backwards_rasterization", &backwards_rasterization, "3DGS Backwards Rasterization (CUDA)"); 
    m.def("backwards_SH", &backwards_SH, "3DGS backwards Delighting (CUDA)"); 
    m.def("backwards_projection", &backwards_projection, "3DGS Backwards Projection (CUDA)"); 


}
