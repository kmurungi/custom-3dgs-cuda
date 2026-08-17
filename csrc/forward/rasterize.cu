#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


__global__ void count_overlaps(
    int num_gaussians,
    const float2* __restrict__ mean_2d_ptr, // N x 2
    const float* __restrict__ cov_2d_ptr,   // N x 3 upper-triangular [a, b, c]
    int width,
    int height,
    int* __restrict__ num_overlap_ptr       // N
){
    // create bounding boxes (1 thread per gaussian)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    float2 mean = mean_2d_ptr[idx];

    // Unpack upper triangular covariance matrix [a, b, c]
    float a = cov_2d_ptr[idx * 3 + 0];
    float b = cov_2d_ptr[idx * 3 + 1];
    float c = cov_2d_ptr[idx * 3 + 2];

    // eigenvalues
    float det = a * c - b * b;
    if (det <= 0.0f) {
        num_overlap_ptr[idx] = 0;
        return;
    }

    float mid = 0.5f * (a + c);
    float lambda_max = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    float radius = ceilf(3.0f * sqrtf(lambda_max));

    // Pixel bounds
    int min_x = max(0, (int)floorf(mean.x - radius));
    int max_x = min(width, (int)ceilf(mean.x + radius));
    int min_y = max(0, (int)floorf(mean.y - radius));
    int max_y = min(height, (int)ceilf(mean.y + radius));

    // Tile grid size
    int tiles_x = (width + 15) / 16;
    int tiles_y = (height + 15) / 16;

    // Tile bounds (16x16), max is exclusive
    int min_tile_x = min_x / 16;
    int max_tile_x = (max_x + 15) / 16;
    int min_tile_y = min_y / 16;
    int max_tile_y = (max_y + 15) / 16;

    // Clamp to tile grid bounds
    min_tile_x = max(0, min(tiles_x, min_tile_x));
    max_tile_x = max(0, min(tiles_x, max_tile_x));
    min_tile_y = max(0, min(tiles_y, min_tile_y));
    max_tile_y = max(0, min(tiles_y, max_tile_y));

    int count = max(0, max_tile_x - min_tile_x) * max(0, max_tile_y - min_tile_y);
    num_overlap_ptr[idx] = count;
}

__global__ void key_generation(
    int num_gaussians,
    const float2* __restrict__ mean_2d_ptr,
    const float* __restrict__ cov_2d_ptr,   // N x 3 upper-triangular [a, b, c]
    const float* __restrict__ depths_ptr,
    const int* __restrict__ offset_ptr,
    const int* __restrict__ num_overlaps_ptr,
    int width,
    int height,
    uint64_t* __restrict__ unsorted_keys_ptr,
    int* __restrict__ unsorted_ids_ptr
){
    // 1 thread per gaussian
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    int num_overlaps = num_overlaps_ptr[idx];
    if (num_overlaps == 0) return;

    int current_slot = offset_ptr[idx];
    float depth = depths_ptr[idx];
    uint32_t depth_as_uint = __float_as_uint(depth);

    // recompute the tile bounding box
    float2 mean = mean_2d_ptr[idx];

    // Unpack upper triangular covariance matrix [a, b, c]
    float a = cov_2d_ptr[idx * 3 + 0];
    float b = cov_2d_ptr[idx * 3 + 1];
    float c = cov_2d_ptr[idx * 3 + 2];

    // eigenvalues
    float det = a * c - b * b;
    if (det <= 0.0f) return;

    float mid = 0.5f * (a + c);
    float lambda_max = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    float radius = ceilf(3.0f * sqrtf(lambda_max));

    // Pixel bounds
    int min_x = max(0, (int)floorf(mean.x - radius));
    int max_x = min(width, (int)ceilf(mean.x + radius));
    int min_y = max(0, (int)floorf(mean.y - radius));
    int max_y = min(height, (int)ceilf(mean.y + radius));

    // Tile grid size
    int tiles_x = (width + 15) / 16;
    int tiles_y = (height + 15) / 16;

    // Tile bounds (16x16), max is exclusive
    int min_tile_x = min_x / 16;
    int max_tile_x = (max_x + 15) / 16;
    int min_tile_y = min_y / 16;
    int max_tile_y = (max_y + 15) / 16;

    // Clamp to tile grid bounds
    min_tile_x = max(0, min(tiles_x, min_tile_x));
    max_tile_x = max(0, min(tiles_x, max_tile_x));
    min_tile_y = max(0, min(tiles_y, min_tile_y));
    max_tile_y = max(0, min(tiles_y, max_tile_y));

    for (int ty = min_tile_y; ty < max_tile_y; ty++) {
        for (int tx = min_tile_x; tx < max_tile_x; tx++) {
            uint64_t tile_id = (uint64_t)(ty * tiles_x + tx);
            uint64_t key = (tile_id << 32) | (uint64_t)depth_as_uint;

            // Write directly to global VRAM
            unsorted_keys_ptr[current_slot] = key;
            unsorted_ids_ptr[current_slot] = idx; // Original Gaussian ID

            current_slot++;
        }
    }
}



__global__ void identify_tile_ranges_kernel(
    int total_duplicated_pairs, // M
    const uint64_t* __restrict__ sorted_keys, // M
    int2* __restrict__ tile_ranges
){
    // 1 thread per duplicated pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_duplicated_pairs) return;

    uint32_t curr_tile = (uint32_t)(sorted_keys[idx] >> 32);

    // if first pair, mark start of its tile
    if (idx == 0) {
        tile_ranges[curr_tile].x = 0;
    } else {
        uint32_t prev_tile = (uint32_t)(sorted_keys[idx - 1] >> 32);
        if (curr_tile != prev_tile) {
            tile_ranges[prev_tile].y = idx; // end
            tile_ranges[curr_tile].x = idx; // start
        }
    }

    // if last pair, mark end of its tile
    if (idx == total_duplicated_pairs - 1) {
        tile_ranges[curr_tile].y = total_duplicated_pairs;
    }
}

__global__ void tile_based_rasterization(
    int height, int width,
    int tiles_x, int tiles_y,
    const int2* __restrict__ tile_ranges,
    const int* __restrict__ sorted_ids,
    const float2* __restrict__ means_2d,
    const float3* __restrict__ cov_2d,      // N x 3 covariance [a, b, c]
    const float3* __restrict__ colors,
    const float* __restrict__ alphas,
    float3* __restrict__ out_img
){
    constexpr int BLOCK_SIZE = 256; // 16x16 tile

    // tile coordinates
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_id = tile_y * tiles_x + tile_x;

    int tile_start = tile_ranges[tile_id].x;
    int tile_end = tile_ranges[tile_id].y;
    int K = tile_end - tile_start;

    // Pixel coordinates
    int px = tile_x * 16 + threadIdx.x;
    int py = tile_y * 16 + threadIdx.y;
    int pixel_idx = py * width + px;

    // flattened thread id within tile
    int thread_id_1d = threadIdx.y * 16 + threadIdx.x;

    bool inside_screen = (px < width && py < height);

    // allocate shared memory (distinct names from global params)
    __shared__ float2 s_means[BLOCK_SIZE];
    __shared__ float3 s_conics[BLOCK_SIZE];
    __shared__ float3 s_colors[BLOCK_SIZE];
    __shared__ float s_opacity[BLOCK_SIZE];

    // default
    float3 pix_color = make_float3(0.0f, 0.0f, 0.0f);
    float T = 1.0f; // Transmittance starts at 100%
    bool done = !inside_screen;

    for (int progress = 0; progress < K; progress += BLOCK_SIZE) {
        int remaining = K - progress;
        int to_load = min(BLOCK_SIZE, remaining);

        // pull from global VRAM via sorted gaussian ids
        if (thread_id_1d < to_load) {
            int gaussian_id = sorted_ids[tile_start + progress + thread_id_1d];
            s_means[thread_id_1d] = means_2d[gaussian_id];

            float3 cov = cov_2d[gaussian_id];
            float det = cov.x * cov.z - cov.y * cov.y;
            float det_inv = (det > 0.0f) ? (1.0f / det) : 0.0f;
            // conic = inverse covariance
            s_conics[thread_id_1d] = make_float3(
                cov.z * det_inv,
                -cov.y * det_inv,
                cov.x * det_inv
            );

            s_colors[thread_id_1d] = colors[gaussian_id];
            s_opacity[thread_id_1d] = alphas[gaussian_id];
        }

        __syncthreads();

        // blend loaded batch into this pixel
        if (!done) {
            for (int i = 0; i < to_load; i++) {
                float2 mean = s_means[i];
                float dx = (float)px - mean.x;
                float dy = (float)py - mean.y;

                float3 conic = s_conics[i];
                float power = -0.5f * (
                    conic.x * dx * dx + 2.0f * conic.y * dx * dy + conic.z * dy * dy
                );

                if (power > 0.0f) continue; // outside influence

                float alpha = fminf(0.99f, s_opacity[i] * __expf(power));
                if (alpha < (1.0f / 255.0f)) continue;

                float weight = alpha * T;
                pix_color.x += s_colors[i].x * weight;
                pix_color.y += s_colors[i].y * weight;
                pix_color.z += s_colors[i].z * weight;

                T *= (1.0f - alpha);
                if (T < 1e-4f) {
                    done = true;
                    break;
                }
            }
        }

        // Exit early if every pixel in the tile is done
        if (__syncthreads_count(done) == BLOCK_SIZE) break;
    }

    if (inside_screen) {
        out_img[pixel_idx] = pix_color;
    }
}


torch::Tensor launch_rasterization(
    int num_gaussians, // N
    const torch::Tensor depths, // N x 1 (needed for depth)
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 3 upper-triangular [a, b, c]
    const torch::Tensor colors,  // N x 3
    const torch::Tensor alpha, // N x 1
    int height,
    int width
){
    auto device = mean_2d.device();
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Early exit: no gaussians
    if (num_gaussians == 0) {
        return torch::zeros({height, width, 3}, float_options);
    }

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

    // shape checks — cov must be packed upper-triangular (N, 3)
    TORCH_CHECK(mean_2d.dim() == 2 && mean_2d.size(0) == num_gaussians && mean_2d.size(1) == 2,
                "mean_2d must have shape (N, 2)");
    TORCH_CHECK(cov_2d.dim() == 2 && cov_2d.size(0) == num_gaussians && cov_2d.size(1) == 3,
                "cov_2d must have shape (N, 3) upper-triangular [a, b, c]");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == num_gaussians && colors.size(1) == 3,
                "colors must have shape (N, 3)");
    TORCH_CHECK(depths.numel() == num_gaussians, "depths must have N elements");
    TORCH_CHECK(alpha.numel() == num_gaussians, "alpha must have N elements");

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
    COUNT OVERLAPS: KERNEL 1 
    */////////////////////////////
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto key_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor num_overlap = torch::zeros({num_gaussians}, int_options); 
    int* num_overlap_ptr = num_overlap.data_ptr<int>();

    count_overlaps<<<blocks, threads>>>(
        num_gaussians,
        mean_2d_ptr,
        cov_2d_ptr,
        width,
        height,
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
    KEY GENERATION + DUPLICATION: KERNEL 2 
    */////////////////////////////
    int last_offset = 0;
    int last_count = 0;

    // mem back to cpu (safe: num_gaussians > 0 guaranteed above)
    cudaMemcpy(
        &last_offset, offset_ptr + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &last_count, num_overlap_ptr + num_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total_duplicated_pairs = last_offset + last_count;

    // Early exit: no tile overlaps
    if (total_duplicated_pairs == 0) {
        return torch::zeros({height, width, 3}, float_options);
    }

    torch::Tensor unsorted_keys = torch::zeros({total_duplicated_pairs}, key_options);
    torch::Tensor unsorted_ids = torch::zeros({total_duplicated_pairs}, int_options); 
    int64_t* unsorted_keys_ptr = unsorted_keys.data_ptr<int64_t>();
    int* unsorted_ids_ptr = unsorted_ids.data_ptr<int>();

    key_generation<<<blocks, threads>>>(
        num_gaussians,
        mean_2d_ptr,
        cov_2d_ptr,
        depths_ptr,
        offset_ptr,
        num_overlap_ptr,
        width,
        height,
        reinterpret_cast<uint64_t*>(unsorted_keys_ptr),
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
    */////////////////////////////

    // tile based parallelization: 16 x 16 pixel tiles
    dim3 render_threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 render_blocks(
        (width + render_threads.x - 1) / render_threads.x,
        (height + render_threads.y - 1) / render_threads.y, 
        1
     );
    // rendered img
    torch::Tensor rendered_img = torch::zeros({height, width, 3}, float_options);
    float3* rendered_img_ptr  = reinterpret_cast<float3*>(rendered_img.data_ptr<float>());
    
    tile_based_rasterization<<<render_blocks, render_threads>>>(
        height, width,
        tiles_x, tiles_y,
        tile_ranges_ptr,
        sorted_ids.data_ptr<int>(),
        mean_2d_ptr,
        reinterpret_cast<const float3*>(cov_2d_ptr),
        colors_ptr,
        alpha_ptr,
        rendered_img_ptr
    ); 

    return rendered_img; 
}