/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

using torch::Tensor;

namespace grouped_gemm {

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

template <typename T>
__global__ void moe_permute_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   int *map_source_row_to_dest_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    if (threadIdx.x == 0)
    {
        // write the map for the following unpermuting
        map_source_row_to_dest_row[source_row] = dest_row;
    }

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x; tid < num_cols; tid += blockDim.x)
    {
        dest_row_ptr[tid] = source_row_ptr[tid];
    }
}

template <typename T>
__global__ void moe_recover_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *map_dest_row_to_source_row,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = map_dest_row_to_source_row[dest_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x; tid < num_cols; tid += blockDim.x)
    {
        dest_row_ptr[tid] = source_row_ptr[tid];
    }
}

template <typename T>
void moe_permute_kernel_launcher(
    const T *original_input,
    T *permuted_output,
    const int *map_dest_row_to_source_row,
    int *map_source_row_to_dest_row,
    const int num_rows,
    const int num_cols,
    cudaStream_t stream)
{
    const int blocks = num_rows;
    const int threads = std::min(num_cols, 1024);

    if (map_source_row_to_dest_row != nullptr)
    {
        moe_permute_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              map_source_row_to_dest_row,
                                                              num_rows,
                                                              num_cols);
    }
    else
    {
        moe_recover_kernel<T><<<blocks, threads, 0, stream>>>(original_input,
                                                              permuted_output,
                                                              map_dest_row_to_source_row,
                                                              num_rows,
                                                              num_cols);
    }
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<Tensor>> moe_permute_op(
    Tensor original_input,
    Tensor expert_for_rows,
    std::vector<Tensor> workspace,
    int64_t max_token_num)
{
    const int num_rows = original_input.size(0);
    const int num_cols = original_input.size(1);

    // activations type
    const at::ScalarType _st = original_input.scalar_type();

    // initialize the workspace on the first run
    if (workspace.empty()) {
        // printf("Permute op workspace initialized!\n");

        auto options = torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(torch::kCUDA)
                        .requires_grad(false);
        Tensor row_id = torch::range(0, max_token_num - 1, 1, options);
        Tensor sorted_expert_for_rows = torch::empty(max_token_num, options);
        Tensor dest_row_to_source_row = torch::empty(max_token_num, options);
        Tensor permuted_output =
            torch::empty({max_token_num, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(row_id);
        workspace.push_back(sorted_expert_for_rows);
        workspace.push_back(dest_row_to_source_row);
        workspace.push_back(permuted_output);
    }

    int *expert_for_rows_ptr = get_ptr<int>(expert_for_rows);
    int *row_id_ptr = get_ptr<int>(workspace[0]);
    int *sorted_expert_for_rows_ptr = get_ptr<int>(workspace[1]);
    int *dest_row_to_source_row_ptr = get_ptr<int>(workspace[2]);
    Tensor permuted_output = workspace[3].narrow(0, 0, num_rows);

    // Run sorting operation
    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    expert_for_rows_ptr, sorted_expert_for_rows_ptr,
                                    row_id_ptr, dest_row_to_source_row_ptr, num_rows);

    int *&map_dest_row_to_source_row = dest_row_to_source_row_ptr;
    Tensor &source_row_to_dest_row = workspace[1];
    int *&map_source_row_to_dest_row = sorted_expert_for_rows_ptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return std::make_tuple(permuted_output, source_row_to_dest_row, workspace);
}

std::tuple<torch::Tensor, std::vector<Tensor>> moe_recover_op(
    Tensor permuted_input,
    Tensor source_row_to_dest_row,
    std::vector<Tensor> workspace,
    int64_t max_token_num)
{
    const int num_rows = permuted_input.size(0);
    const int num_cols = permuted_input.size(1);

    // activations type
    const at::ScalarType _st = permuted_input.scalar_type();

    // initialize the workspace on the first run
    if (workspace.empty()) {
        // printf("Permute op backward workspace initialized!\n");

        Tensor original_output =
            torch::empty({max_token_num, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(original_output);
    }

    Tensor original_output = workspace[0].narrow(0, 0, num_rows);

    int *map_source_row_to_dest_row = get_ptr<int>(source_row_to_dest_row);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return std::make_tuple(original_output, workspace);
}


}  // namespace grouped_gemm
