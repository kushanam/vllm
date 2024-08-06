/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "penaltyKernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"


void batch_apply_penalty(
    torch::Tensor& input_logits,      
    torch::Tensor&  output_logits,
    std::int32_t max_seq_len, std::int32_t vocab_size,
    torch::Tensor& penalty_workspace, torch::Tensor& penalty_workspace_prev, torch::Tensor& temperatures,
    torch::Tensor& repetition_penalties, torch::Tensor& presence_penalties, torch::Tensor& frequency_penalties,
    torch::Tensor& output_Ids, torch::Tensor& parent_Ids, torch::Tensor& input_lengths,
    torch::Tensor& sequence_lengths, 
    torch::Tensor& tokens_per_step,
    std::int32_t const batch_size, std::int32_t const beam_width, std::int32_t const maxtokens_per_step)
{
  CUmodule cuModule;
  cuModuleLoad(&cuModule, "penaltyKernels.cubin");
  CUfunction penaltyFunc;
  cuModuleGetFunction(&penaltyFunc, cuModule, "batchApplyPenaltyKernel");
  void* args[] = {&input_logits.data_ptr<scalar_t>(),
        &output_logits.data_ptr<scalar_t>(),
        &max_seq_len,
        &vocab_size,
        &penalty_workspace.data_ptr<int>(), 
        &penalty_workspace_prev.data_ptr<int>(), 
        &temperatures.data_ptr<float>(),
        &repetition_penalties.data_ptr<float>(),
        &presence_penalties.data_ptr<float>(), 
        &frequency_penalties.data_ptr<float>(),
        &output_Ids.data_ptr<int>(),
        &parent_Ids.data_ptr<int>(), 
        &input_lengths.data_ptr<int>(),
        &sequence_lengths.data_ptr<int>(),
        &tokens_per_step.data_ptr<int>())
         };
  dim3 block(256);
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(input_logits));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  cuLaunchKernel(cuFunction,
                             batch_size, 1, 1,
                             beam_width, 1, 1,
                             0, 0, args, 0);
  CUDA_CALL(cuModuleUnload(cuModule));
}

