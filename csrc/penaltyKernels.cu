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

template <typename scalar_t>
__global__ void batchApplyPenaltyKernel(scalar_t* input_logits, scalar_t* output_logits,
    std::int32_t max_seq_len, std::int32_t vocab_size,
    float const* temperatures,
    float const* repetition_penalties, float const* presence_penalties, float const* frequency_penalties,
    std::int32_t const* sequence_lengths)
{
    auto const inLogitsPtr = input_logits +  blockIdx.x * vocab_size;
    auto outLogitsPtr = output_logits + blockIdx.x * vocab_size;
    const scalar_t MASK_VAL = (std::is_same<scalar_t, half>::value) ? -HALF_FLT_MAX : -FLT_MAX;
    float invTemperature, repetition_penalty, presence_penalty, frequency_penalty;
    if (temperatures != nullptr)
    {
        invTemperature = 1.0f / (temperatures[batchSlot] + 1e-6f);
    }
    if (repetition_penalties != nullptr)
    {
        repetition_penalty = repetition_penalties[batchSlot];
    }
    if (presence_penalties != nullptr)
    {
        presence_penalty = presence_penalties[batchSlot];
    }
    if (frequency_penalties != nullptr)
    {
        frequency_penalty = frequency_penalties[batchSlot];
    }
    for (auto index = static_cast<std::int32_t>(threadIdx.x); index < vocab_size;
         index += static_cast<std::int32_t>(blockDim.x))
    {
        if (index < vocab_size)
        {
            auto logit = static_cast<float>(inLogitsPtr[index]);

            // Temperature
            if (temperatures != nullptr)
            {
                logit *= invTemperature;
            }

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
                logit -= frequency_penalty * numOccurences;
            }

            outLogitsPtr[index] = logit;
        }
        else
        {
            outLogitsPtr[index] = MASK_VAL;
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
    "batchApplyPenaltyKernel",                                                                 \
    [&] {                                                                                      \
      vllm::batchApplyPenaltyKernel<scalar_t><<<grid, block, 0, stream>>>(                     \
        input_logits.data_ptr<scalar_t>(),                                                      \
        output_logits.data_ptr<scalar_t>(),                                                     \
        batch_size, vocab_size,                                                 \
        temperatures.data_ptr<float>(),                                  \
        repetition_penalties.data_ptr<float>(),                   \
        presence_penalties.data_ptr<float>(),                  \
        frequency_penalties.data_ptr<float>(),          \
        sequence_lengths.data_ptr<std::int32_t>() );                                              \
    });


void batchApplyPenalty(
    torch::Tensor& input_logits,      
    torch::Tensor&  output_logits,
    std::int32_t const batch_size, std::int32_t vocab_size,
    float const* temperatures,
    float const* repetition_penalties, float const* presence_penalties, float const* frequency_penalties,
    std::int32_t const* sequence_lengths)   
{
  LAUNCH_PENALTY_KERNEL();
}




