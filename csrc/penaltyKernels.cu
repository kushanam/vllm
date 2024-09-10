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


namespace vllm
{
constexpr float HALF_MAX = 65504.f;
template <typename scalar_t>
__global__ void batchApplyPenaltyKernel(scalar_t* input_logits, scalar_t* output_logits,
    std::int64_t batch_size, std::int64_t vocab_size, std::int64_t max_seq_len,
    std::int32_t* penalty_workspace,
    float const* temperatures,
    float const* repetition_penalties, float const* presence_penalties, float const* frequency_penalties,
    std::int32_t const* output_ids,
    std::int32_t const* sequence_lengths,
    std::int32_t const* aggregate_lengths)
{
    int32_t batch_pos = blockIdx.x;
    auto const in_logits_ptr = input_logits +  blockIdx.x * vocab_size;
    auto out_logits_ptr = output_logits + blockIdx.x * vocab_size;
    const scalar_t MASK_VAL = (std::is_same<scalar_t, half>::value) ? -HALF_MAX : -FLT_MAX;
    float invTemperature, repetition_penalty, presence_penalty, frequency_penalty;

    auto const input_len = sequence_lengths[batch_pos];
    auto const current_step = aggregate_lengths[batch_pos];

    penalty_workspace += batch_pos * vocab_size;
    if (current_step <= input_len)
    { // Context phase
        for (auto index = static_cast<std::int32_t>(threadIdx.x); index < vocab_size;
                index += static_cast<std::int32_t>(blockDim.x))
        {
            penalty_workspace[index] = 0;
        }
        __syncthreads();
        for (auto step = static_cast<std::int32_t>(threadIdx.x); step < input_len;
                step += static_cast<std::int32_t>(blockDim.x))
        {
            auto penalty_index = output_ids[batch_pos * max_seq_len + step];
            if (penalty_index < vocab_size)
            {
                atomicAdd(&penalty_workspace[penalty_index], 1);
            }
        }
    }
    else
    { // Generation phase

        if (threadIdx.x == 0)
        {
            auto penalty_index = output_ids[batch_pos * max_seq_len + current_step - 1];
            if (penalty_index < vocab_size)
            {
                penalty_workspace[penalty_index] += 1;
            }
        }
    }
    __syncthreads();

    if (temperatures != nullptr)
    {
        invTemperature = 1.0f / (temperatures[batch_pos] + 1e-6f);
    }
    if (repetition_penalties != nullptr)
    {
        repetition_penalty = repetition_penalties[batch_pos];
    }
    if (presence_penalties != nullptr)
    {
        presence_penalty = presence_penalties[batch_pos];
    }
    if (frequency_penalties != nullptr)
    {
        frequency_penalty = frequency_penalties[batch_pos];
    }
    for (auto index = static_cast<std::int32_t>(threadIdx.x); index < vocab_size;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        if (index < vocab_size)
        {
            auto logit = static_cast<float>(in_logits_ptr[index]);

            // Temperature
            if (temperatures != nullptr)
            {
                logit *= invTemperature;
            }
            std::int32_t num_occurences = penalty_workspace[index];
            if (num_occurences > 0)
            {
                // Repetition
                if (repetition_penalties != nullptr)
                {
                    logit = logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty;
                }
                // Presence
                if (presence_penalties != nullptr)
                {
                    logit -= presence_penalty;
                }
                // Frequency
                if (frequency_penalties != nullptr)
                {
                    logit -= frequency_penalty * num_occurences;
                }
            }
            out_logits_ptr[index] = logit;
        }
        else
        {
            out_logits_ptr[index] = MASK_VAL;
        }
    }
}

}
#define LAUNCH_PENALTY_KERNEL()                                                           \
  dim3 block(256);                                                                        \
  dim3 grid(batch_size, 1, 1);                                                            \
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(input_logits));               \
  auto stream = c10::cuda::getCurrentCUDAStream().stream();                               \
  VLLM_DISPATCH_FLOATING_TYPES(                                                           \
    input_logits.scalar_type(),                                                           \
    "batchApplyPenaltyKernel",                                                            \
    [&] {                                                                                 \
      vllm::batchApplyPenaltyKernel<scalar_t><<<grid, block, 0, stream>>>(                \
        input_logits.data_ptr<scalar_t>(),                                                \
        output_logits.data_ptr<scalar_t>(),                                               \
        batch_size, vocab_size, max_seq_len,                                              \
        penalty_workspace.data_ptr<std::int32_t>(),                                       \
        temperatures.data_ptr<float>(),                                                   \
        repetition_penalties.data_ptr<float>(),                                           \
        presence_penalties.data_ptr<float>(),                                             \
        frequency_penalties.data_ptr<float>(),                                            \
        output_ids.data_ptr<std::int32_t>(),                                              \
        sequence_lengths.data_ptr<std::int32_t>(),                                        \
        aggregate_lengths.data_ptr<std::int32_t>());                                      \
    });


void batch_apply_penalty(
    torch::Tensor& input_logits,      
    torch::Tensor&  output_logits,
    std::int64_t const batch_size, std::int64_t vocab_size, std::int64_t max_seq_len,
    torch::Tensor&  penalty_workspace,
    torch::Tensor& temperatures, torch::Tensor& repetition_penalties,
    torch::Tensor& presence_penalties, torch::Tensor& frequency_penalties,
    torch::Tensor&  output_ids,
    torch::Tensor& sequence_lengths,
    torch::Tensor& aggregate_lengths)   
{
  LAUNCH_PENALTY_KERNEL();
}



