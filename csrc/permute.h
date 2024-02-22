/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <torch/extension.h>

using torch::Tensor;

namespace grouped_gemm {

// std::tuple<torch::Tensor, torch::Tensor> moe_permute_op(torch::Tensor original_input,torch::Tensor expert_for_rows);
std::tuple<torch::Tensor, torch::Tensor, std::vector<Tensor>> moe_permute_op(
    Tensor original_input,
    Tensor expert_for_rows,
    std::vector<Tensor> workspace,
    int64_t max_token_num);

// torch::Tensor moe_recover_op(torch::Tensor permuted_input, torch::Tensor source_row_to_dest_row);

std::tuple<torch::Tensor, std::vector<Tensor>> moe_recover_op(
    Tensor permuted_input,
    Tensor source_row_to_dest_row,
    std::vector<Tensor> workspace,
    int64_t max_token_num);

}  // namespace grouped_gemm