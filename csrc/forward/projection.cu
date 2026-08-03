#include <torch/extension.h>
#include "projection.h"


__device__ inline 

__device__ inline 2by2matmul(){

}

__device__ inline 3by3matmul(){ 

    return 0
}

__global__ void projection(){ 

}


void run_projection(
    const torch::tensor& mu,
    const torch::tensor& q,
    const torch::tensor& s,
    const torch::tensor& img,
){
    return projection<<<numBlocks, numThreadsPerBlock>>>(mu, q, s, img);

}