#include <torch/extension.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cuda_runtime.h> 
#include <cub/cub.cuh> 

__global__ void tile_based_rasterization(
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 2 x 2
    const torch::Tensor colors,  // N x 3 
    const torch::Tensor alpha // N x 1 
    int height, 
    int width, 
    float2* rendered_img_ptr // H x W

){




}


torch::Tensor launch_rasterization(
    int num_gaussians, // N 
    const torch::Tensor depths, // N x 1 (needed for depth)
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 2 x 2
    const torch::Tensor colors,  // N x 3 
    const torch::Tensor alpha, // N x 1 
    int height, 
    int width
){
    // cuda tensor checks
    TORCH_CHECK(depths.is_cuda(), "depths must be a CUDA tensor");
    TORCH_CHECK(mean_2d.is_cuda(), "2D means must be a CUDA tensor");
    TORCH_CHECK(cov_2d.is_cuda(), "2D Cov must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "Alpha must be a CUDA tensor");

    // contiguous checks
    TORCH_CHECK(depths.is_contiguous(), "depths must be contiguous in memory");
    TORCH_CHECK(mean_2d.is_contiguous(), "2D means must be contiguous in memory");
    TORCH_CHECK(cov_2d.is_contiguous(), "2D Cov must be contiguous in memory");
    TORCH_CHECK(colors.is_contiguous(), "colors must be contiguous in memory");
    TORCH_CHECK(alpha.is_contiguous(), "Alpha must be contiguous in memory");

    // extract float pointer from tensor
    const float* depths_ptr = depths.data_ptr<float>();
    const float2* mean_2d_ptr = reinterpret_cast<const float2*>(mean_2d.data_ptr<float>());
    const float* cov_2d_ptr = cov_2d.data_ptr<float>();
    const float3* colors_ptr = reinterpret_cast<const float3*>(colors.data_ptr<float>());
    const float* alpha_ptr = alpha.data_ptr<float>();


    // 1 gaussian per thread
    int threadsPerBlock = 256; 
    int blocks = (num_gaussians + threadsPerBlock - 1)/threadsPerBlock; 

    
    /*////////////////////////////
    COUNT OVERLAPS 
    */////////////////////////////
    auto device = mean_2d.device(); 
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor num_overlap = torch::zeros({num_gaussians}, options); 
    float* num_overlap_ptr = num_overlap.data_ptr<float>();

    count_overlaps<<<blocks, threadsPerBlock>>>(
        mean_2d,
        cov_2d, 
        num_overlap_ptr
    ); 
    
    /*////////////////////////////
    PREFIX ARRAY + CREATION OF BUFFER ARRAY
    */////////////////////////////
    torch::Tensor offset = torch::zeros({num_gaussians}, options); 
    float* offset_ptr = offset.data_ptr<float>();
    size_t temp_storage_bytes = 0; 
    
    cub::DeviceScan::ExclusiveSum( 
        nullptr, temp_storage_bytes, 
        count_overlap_ptr, offset_ptr, num_gaussians
    ); // prefix sum returned in offset

    /*////////////////////////////
    KEY GENERATION + DUPLICATION
    */////////////////////////////
    int total_duplicated_pairs = offset[-1] + num_overlap_ptr[-1]; 

    torch::Tensor unsorted_keys = torch::zeros({total_duplicated_pairs}, options); 
    torch::Tensor unsorted_ids = torch::zeros({total_duplicated_pairs}, options); 

    key_generation<<<blocks, threadsPerBlock>>>(
        num_gaussians, 
        mean_2d_ptr,
        depths_ptr, 
        offset_ptr, 
        unsorted_keys, 
        unsorted_ids
    ); 

    /*////////////////////////////
    RADIX SORT
    */////////////////////////////
    auto sorted_keys = torch::empty({total_duplicated_pairs}, options); 
    auto sorted_ids = torch::empty({total_duplicated_pairs}, options);

    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(unsorted_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        unsorted_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
        total_duplicated_pairs
    );
    auto temp_buffer2 = torch::empty({(long)temp_storage_bytes}, options);
    cub::DeviceRadixSort::SortPairs(
        temp_buffer2.data_ptr(), temp_storage_bytes,
        reinterpret_cast<uint64_t*>(unsorted_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        unsorted_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(),
        total_duplicated_pairs
    );


    /*////////////////////////////
    TILE RANGE 
    */////////////////////////////
    auto tile_ranges = torch::zeros({total_tiles, 2}, options); 
    int blocks_m = (total_duplicated_pairs + threads - 1) / threads; 
    
    identify_tile_ranges_kernel<<<blocks_m, threads>>>(
        total_duplicated_pairs,
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        reinterpret_cast<int2*>(tile_ranges.data_ptr<int>())
    ); 

    /*////////////////////////////
    FINAL RENDERING 
    *////////////////////////////

    // tile based parallelization: 16 x 16 pixel tiles
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
         (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
         (height + threadsPerBlock.y - 1) / threadsPerBlock.y
     );
    // rendered img 
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mean_2d.device());
    torch::Tensor rendered_img = torch::zeros({height, width, 3}, options);
    float3* rendered_img_ptr  = reinterpret_cast<float3*>(rendered_img.data_ptr<float>());
    
    tile_based_rasterization<<<numBlocks, threadsPerBlock>>>(
        num_gaussians, // N
        mean_2d_ptr, // N x 2
        cov_2d_ptr,  // N x 2 x 2
        colors_ptr,  // N x 3 
        alpha_ptr,  // N x 1 
        height, width, 
        tile_ranges, 
        rendered_img_ptr // Height x Width
    ); 

    return rendered_img; 
}