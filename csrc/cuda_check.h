#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Runtime status from a CUDA API or CUB call. Parenthesize expressions that
// contain commas: CUDA_CHECK((cub::DeviceScan::ExclusiveSum(...))).
inline void cuda_check_impl(cudaError_t status, const char* expr, const char* file, int line) {
    TORCH_CHECK(
        status == cudaSuccess,
        "CUDA error: ",
        cudaGetErrorString(status),
        " (",
        expr,
        ") at ",
        file,
        ":",
        line);
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

#define CHECK_CUDA(tensor) \
    TORCH_CHECK((tensor).is_cuda(), #tensor " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(tensor) \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")

#define CHECK_FP32(tensor) \
    TORCH_CHECK((tensor).scalar_type() == torch::kFloat32, #tensor " must be FP32")

#define CHECK_INT32(tensor) \
    TORCH_CHECK((tensor).scalar_type() == torch::kInt32, #tensor " must be int32")

#define CHECK_INPUT_FP32(tensor) \
    do {                         \
        CHECK_CUDA(tensor);      \
        CHECK_CONTIGUOUS(tensor);\
        CHECK_FP32(tensor);      \
    } while (0)

#define CHECK_INPUT_INT32(tensor) \
    do {                          \
        CHECK_CUDA(tensor);       \
        CHECK_CONTIGUOUS(tensor); \
        CHECK_INT32(tensor);      \
    } while (0)
