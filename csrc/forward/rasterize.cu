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
    const int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;

    constexpr int TILE_SIZE = 16;
    const int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_tiles = tiles_x * tiles_y;

    
    /*////////////////////////////
    COUNT OVERLAPS 
    */////////////////////////////
    auto device = mean_2d.device(); 
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto key_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor num_overlap = torch::zeros({num_gaussians}, int_options); 
    int* num_overlap_ptr = num_overlap.data_ptr<int>();

    count_overlaps<<<blocks, threads>>>(
        mean_2d_ptr,
        cov_2d_ptr, 
        num_overlap_ptr
    ); 
    
    /*////////////////////////////
    PREFIX ARRAY + CREATION OF BUFFER ARRAY
    */////////////////////////////
    torch::Tensor offset = torch::zeros({num_gaussians}, int_options); 
    int* offset_ptr = offset.data_ptr<int>();
    size_t temp_storage_bytes = 0; 
    
    cub::DeviceScan::ExclusiveSum( 
        nullptr, temp_storage_bytes, 
        num_overlap_ptr, offset_ptr, num_gaussians
    ); // query temp storage size
    auto scan_temp = torch::empty({static_cast<long>(temp_storage_bytes)}, byte_options);
    cub::DeviceScan::ExclusiveSum(
        scan_temp.data_ptr(), temp_storage_bytes,
        num_overlap_ptr, offset_ptr, num_gaussians
    ); // prefix sum returned in offset

    /*////////////////////////////
    KEY GENERATION + DUPLICATION
    */////////////////////////////
    int last_offset = 0;
    int last_count = 0;

    // mem back to cpu
    cudaMemcpy(
        &last_offset, offset_ptr + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &last_count, num_overlap_ptr + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total_duplicated_pairs = last_offset + last_count; 

    torch::Tensor unsorted_keys = torch::zeros({total_duplicated_pairs}, key_options); 
    torch::Tensor unsorted_ids = torch::zeros({total_duplicated_pairs}, int_options); 
    int64_t* unsorted_keys_ptr = unsorted_keys.data_ptr<int64_t>();
    int* unsorted_ids_ptr = unsorted_ids.data_ptr<int>();

    key_generation<<<blocks, threads>>>(
        num_gaussians, 
        mean_2d_ptr,
        depths_ptr, 
        offset_ptr, 
        unsorted_keys_ptr, 
        unsorted_ids_ptr
    ); 

    /*////////////////////////////
    RADIX SORT
    */////////////////////////////
    auto sorted_keys = torch::empty({total_duplicated_pairs}, key_options); 
    auto sorted_ids = torch::empty({total_duplicated_pairs}, int_options);

    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(unsorted_keys_ptr),
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        unsorted_ids_ptr, sorted_ids.data_ptr<int>(),
        total_duplicated_pairs
    );
    auto sort_temp = torch::empty({static_cast<long>(temp_storage_bytes)}, byte_options);
    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr(), temp_storage_bytes,
        reinterpret_cast<uint64_t*>(unsorted_keys_ptr),
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        unsorted_ids_ptr, sorted_ids.data_ptr<int>(),
        total_duplicated_pairs
    );


    /*////////////////////////////
    TILE RANGE 
    */////////////////////////////
    auto tile_ranges = torch::zeros({total_tiles, 2}, int_options); 
    int2* tile_ranges_ptr = reinterpret_cast<int2*>(tile_ranges.data_ptr<int>());
    int blocks_m = (total_duplicated_pairs + threads - 1) / threads; 
    
    identify_tile_ranges_kernel<<<blocks_m, threads>>>(
        total_duplicated_pairs,
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        tile_ranges_ptr
    ); 

    /*////////////////////////////
    FINAL RENDERING 
    *////////////////////////////

    // tile based parallelization: 16 x 16 pixel tiles
    dim3 render_threads(TILE_SIZE, TILE_SIZE);
    dim3 render_blocks(
        (width + render_threads.x - 1) / render_threads.x,
        (height + render_threads.y - 1) / render_threads.y
     );
    // rendered img 
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor rendered_img = torch::zeros({height, width, 3}, float_options);
    float3* rendered_img_ptr  = reinterpret_cast<float3*>(rendered_img.data_ptr<float>());
    
    tile_based_rasterization<<<render_blocks, render_threads>>>(
        num_gaussians, // N
        mean_2d_ptr, // N x 2
        cov_2d_ptr,  // N x 2 x 2
        colors_ptr,  // N x 3 
        alpha_ptr,  // N x 1 
        height, width, 
        tile_ranges_ptr, 
        rendered_img_ptr // Height x Width
    ); 

    return rendered_img; 
}