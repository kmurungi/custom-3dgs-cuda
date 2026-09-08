#include "rasterize_backward.h"
#include <cuda_runtime.h>


__global__ void backward_rasterization(
    int num_gaussians,
    int height,
    int width,
    const float3* __restrict__ grad_output,
    const float2* __restrict__ mean_2d,
    const float3* __restrict__ cov_2d,
    const float3* __restrict__ colors,
    const float* __restrict__ alpha,
    const int* __restrict__ sorted_ids,
    const int2* __restrict__ tile_ranges,
    const float* __restrict__ final_T,
    const int* __restrict__ n_contrib,

    // global memory
    float2* __restrict__ dl_mean2d,
    float3* __restrict__ dl_cov2d,
    float3* __restrict__ dl_colors,
    float* __restrict__ dl_alpha
){
    constexpr int BLOCK_SIZE = 256; // 16x16 tile
    const int tiles_x = (width + 15) / 16;

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

    // shared memory — inputs + per-batch gradient buffers (indexed by gaussian slot k)
    __shared__ int s_ids[BLOCK_SIZE];
    __shared__ float2 s_means[BLOCK_SIZE];
    __shared__ float3 s_conics[BLOCK_SIZE];
    __shared__ float3 s_colors[BLOCK_SIZE];
    __shared__ float s_opacity[BLOCK_SIZE];

    __shared__ float2 s_dl_means[BLOCK_SIZE];
    __shared__ float3 s_dl_conics[BLOCK_SIZE];
    __shared__ float3 s_dl_colors[BLOCK_SIZE];
    __shared__ float s_dl_opacity[BLOCK_SIZE];

    float3 dL_dC = inside_screen
        ? grad_output[pixel_idx]
        : make_float3(0.0f, 0.0f, 0.0f);
    int last_contributor = inside_screen ? n_contrib[pixel_idx] : 0;

    // back-to-front state (same blending sequence as forward, reversed)
    float T = inside_screen ? final_T[pixel_idx] : 0.0f;
    float3 accum_rec = make_float3(0.0f, 0.0f, 0.0f);
    bool done = !inside_screen || (last_contributor == 0);

    // reverse batches of the sorted list forward walked front-to-back
    for (int progress = 0; progress < K; progress += BLOCK_SIZE) {
        int remaining = K - progress;
        int to_load = min(BLOCK_SIZE, remaining);
        int batch_start = tile_start + K - progress - to_load;

        // pull from GLOBAL via sorted gaussian ids
        if (thread_id_1d < to_load) {
            int gaussian_id = sorted_ids[batch_start + thread_id_1d];
            s_ids[thread_id_1d] = gaussian_id;
            s_means[thread_id_1d] = mean_2d[gaussian_id];

            float3 cov = cov_2d[gaussian_id];
            float det = cov.x * cov.z - cov.y * cov.y;
            float det_inv = (det > 0.0f) ? (1.0f / det) : 0.0f;
            s_conics[thread_id_1d] = make_float3(
                cov.z * det_inv,
                -cov.y * det_inv,
                cov.x * det_inv
            );

            s_colors[thread_id_1d] = colors[gaussian_id];
            s_opacity[thread_id_1d] = alpha[gaussian_id];

            // zero shared grad buffers for this batch slot
            s_dl_means[thread_id_1d] = make_float2(0.0f, 0.0f);
            s_dl_conics[thread_id_1d] = make_float3(0.0f, 0.0f, 0.0f);
            s_dl_colors[thread_id_1d] = make_float3(0.0f, 0.0f, 0.0f);
            s_dl_opacity[thread_id_1d] = 0.0f;
        }

        __syncthreads();

        // back-to-front gradient calculation loop for each gaussian in this batch 
        if (!done) {
            for (int j = 0; j < to_load; j++) {
                int k = to_load - 1 - j;
                int contributor = K - progress - to_load + k + 1; 
                if (contributor > last_contributor) continue;

                float2 mean = s_means[k];
                float dx = (float)px - mean.x;
                float dy = (float)py - mean.y;

                float3 conic = s_conics[k];
                float power = -0.5f * (
                    conic.x * dx * dx + 2.0f * conic.y * dx * dy + conic.z * dy * dy
                );
                if (power > 0.0f) continue; // outside influence

                float opac = s_opacity[k];
                float G = __expf(power);
                float alpha_i = fminf(0.99f, opac * G);
                if (alpha_i < (1.0f / 255.0f)) continue;

                // unwind transmittance to just before this Gaussian
                float one_minus_alpha = 1.0f - alpha_i;
                float T_before = T / one_minus_alpha;

                float3 c = s_colors[k];

                // per-pixel grads for this gaussian
                float3 dL_dcolor = make_float3(
                    alpha_i * T_before * dL_dC.x,
                    alpha_i * T_before * dL_dC.y,
                    alpha_i * T_before * dL_dC.z
                );

                float dL_dalpha =
                    ((c.x - accum_rec.x) * dL_dC.x +
                     (c.y - accum_rec.y) * dL_dC.y +
                     (c.z - accum_rec.z) * dL_dC.z) * T_before;

                float dL_dopacity = G * dL_dalpha;
                float dL_dpower = alpha_i * dL_dalpha;

                float2 dL_dmean = make_float2(
                    dL_dpower * (conic.x * dx + conic.y * dy),
                    dL_dpower * (conic.y * dx + conic.z * dy)
                );

                float3 dL_dconic = make_float3(
                    -0.5f * dL_dpower * dx * dx,
                    -dL_dpower * dx * dy,
                    -0.5f * dL_dpower * dy * dy
                );

                //accumulate into shared slot k
                atomicAdd(&s_dl_means[k].x, dL_dmean.x);
                atomicAdd(&s_dl_means[k].y, dL_dmean.y);

                atomicAdd(&s_dl_conics[k].x, dL_dconic.x);
                atomicAdd(&s_dl_conics[k].y, dL_dconic.y);
                atomicAdd(&s_dl_conics[k].z, dL_dconic.z);

                atomicAdd(&s_dl_colors[k].x, dL_dcolor.x);
                atomicAdd(&s_dl_colors[k].y, dL_dcolor.y);
                atomicAdd(&s_dl_colors[k].z, dL_dcolor.z);

                atomicAdd(&s_dl_opacity[k], dL_dopacity);

                // update reconstructed background color + T for nextb farther gaussian
                accum_rec.x = alpha_i * c.x + one_minus_alpha * accum_rec.x;
                accum_rec.y = alpha_i * c.y + one_minus_alpha * accum_rec.y;
                accum_rec.z = alpha_i * c.z + one_minus_alpha * accum_rec.z;
                T = T_before;

                if (contributor == 1) {
                    done = true;
                    break;
                }
            }
        }

        __syncthreads();

        // flush shared batch grads -> global 
        if (thread_id_1d < to_load) {
            int gid = s_ids[thread_id_1d];
            atomicAdd(&dl_mean2d[gid].x, s_dl_means[thread_id_1d].x);
            atomicAdd(&dl_mean2d[gid].y, s_dl_means[thread_id_1d].y);

            atomicAdd(&dl_cov2d[gid].x, s_dl_conics[thread_id_1d].x);
            atomicAdd(&dl_cov2d[gid].y, s_dl_conics[thread_id_1d].y);
            atomicAdd(&dl_cov2d[gid].z, s_dl_conics[thread_id_1d].z);

            atomicAdd(&dl_colors[gid].x, s_dl_colors[thread_id_1d].x);
            atomicAdd(&dl_colors[gid].y, s_dl_colors[thread_id_1d].y);
            atomicAdd(&dl_colors[gid].z, s_dl_colors[thread_id_1d].z);

            atomicAdd(&dl_alpha[gid], s_dl_opacity[thread_id_1d]);
        }

        if (__syncthreads_count(done) == BLOCK_SIZE) break;
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_rasterization(
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
    constexpr int TILE_SIZE = 16;
    const int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks(tiles_x, tiles_y, 1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mean_2d.device());
    torch::Tensor dl_mean2d = torch::zeros({num_gaussians, 2}, options);
    torch::Tensor dl_cov2d = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor dl_colors = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor dl_alpha = torch::zeros({num_gaussians, 1}, options);

    backward_rasterization<<<blocks, threads>>>(
        num_gaussians,
        height,
        width,
        reinterpret_cast<const float3*>(grad_output.data_ptr<float>()),
        reinterpret_cast<const float2*>(mean_2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(cov_2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(colors.data_ptr<float>()),
        alpha.data_ptr<float>(),
        sorted_ids.data_ptr<int>(),
        reinterpret_cast<const int2*>(tile_ranges.data_ptr<int>()),
        final_T.data_ptr<float>(),
        n_contrib.data_ptr<int>(),
        reinterpret_cast<float2*>(dl_mean2d.data_ptr<float>()),
        reinterpret_cast<float3*>(dl_cov2d.data_ptr<float>()),
        reinterpret_cast<float3*>(dl_colors.data_ptr<float>()),
        dl_alpha.data_ptr<float>()
    );

    return std::make_tuple(dl_mean2d, dl_cov2d, dl_colors, dl_alpha);
}
