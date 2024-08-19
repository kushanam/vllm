/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void batch_apply_penalty(
    torch::Tensor& input_logits,      
    torch::Tensor&  output_logits,
    std::int32_t const batch_size, std::int32_t vocab_size, std::int32_t max_seq_len,
    torch::Tensor&  penalty_workspace,
    torch::Tensor& temperatures, torch::Tensor& repetition_penalties,
    torch::Tensor& presence_penalties, torch::Tensor& frequency_penalties,
    torch::Tensor&  output_ids,
    torch::Tensor& sequence_lengths,
    torch::Tensor& aggregate_lengths);