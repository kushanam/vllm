from typing import List, Optional, Tuple

import pytest
import torch
from vllm import _custom_ops as ops

DTYPES = [torch.float16, torch.bfloat16]

VOCAB_SIZE = 32000
BATCH_SIZES = [16, 32, 64]
MAX_SEQ_LEN = [256]


import torch

def ref_batch_apply_penalty(input_logits, output_logits, batch_size, vocab_size, max_seq_len,
                        penalty_workspace, temperatures=None, repetition_penalties=None,
                        presence_penalties=None, frequency_penalties=None, output_ids=None,
                        sequence_lengths=None, aggregate_lengths=None):
    # Determine the mask value based on the data type
    dtype = input_logits.dtype
    if dtype == torch.float16:
        MASK_VAL = -torch.finfo(torch.float16).max
    else:
        MASK_VAL = -torch.finfo(torch.float32).max

    for batch_pos in range(batch_size):
        in_logits_ptr = input_logits[batch_pos]
        out_logits_ptr = output_logits[batch_pos]

        input_len = sequence_lengths[batch_pos].item()
        current_step = aggregate_lengths[batch_pos].item()

        penalty_workspace[batch_pos].zero_()

        if current_step <= input_len:
            # Context phase
            for step in range(input_len):
                penalty_index = output_ids[batch_pos, step].item()
                if penalty_index < vocab_size:
                    penalty_workspace[batch_pos, penalty_index] += 1
        else:
            # Generation phase
            penalty_index = output_ids[batch_pos, current_step - 1].item()
            if penalty_index < vocab_size:
                penalty_workspace[batch_pos, penalty_index] += 1

        invTemperature = 1.0 / (temperatures[batch_pos].item() + 1e-6) if temperatures is not None else 1.0
        repetition_penalty = repetition_penalties[batch_pos].item() if repetition_penalties is not None else 1.0
        presence_penalty = presence_penalties[batch_pos].item() if presence_penalties is not None else 0.0
        frequency_penalty = frequency_penalties[batch_pos].item() if frequency_penalties is not None else 0.0

        for index in range(vocab_size):
            logit = in_logits_ptr[index].item()

            # Apply temperature scaling
            logit *= invTemperature

            num_occurrences = penalty_workspace[batch_pos, index].item()
            if num_occurrences > 0:
                # Apply repetition penalty
                if logit < 0.0:
                    logit *= repetition_penalty
                else:
                    logit /= repetition_penalty

                # Apply presence penalty
                logit -= presence_penalty

                # Apply frequency penalty
                logit -= frequency_penalty * num_occurrences

            out_logits_ptr[index] = logit if index < vocab_size else MASK_VAL

    return output_logits


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_penalty(
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    
    input_logits = torch.randn(batch_size,
                            MAX_SEQ_LEN,
                            dtype=dtype)
    output_logits = torch.empty_like(input_logits)
    ref_output_logits = torch.empty_like(input_logits)
    workspace = torch.zeros(batch_size, VOCAB_SIZE)
    ref_workspace = torch.zeros(batch_size, VOCAB_SIZE)
    tempratures = torch.randn(batch_size)
    repetition_penalties = torch.randn(batch_size)
    presence_penalties = torch.randn(batch_size)
    frequency_penalties = torch.randn(batch_size)
    sequence_lengths = torch.randint(20, 128, batch_size)
    aggregate_lengths = torch.randint(20, 64, batch_size)
    output_ids = torch.randint(1, VOCAB_SIZE, (batch_size, MAX_SEQ_LEN))

    ops.batch_apply_penalty(input_logits, output_logits,
            batch_size, VOCAB_SIZE, MAX_SEQ_LEN,
            workspace,
            tempratures, repetition_penalties,
            presence_penalties, frequency_penalties,
            output_ids, sequence_lengths, 
            aggregate_lengths)
    
    ref_batch_apply_penalty(input_logits, ref_output_logits,
        batch_size, VOCAB_SIZE, MAX_SEQ_LEN,
        ref_workspace,
        tempratures, repetition_penalties,
        presence_penalties, frequency_penalties,
        output_ids, sequence_lengths, 
        aggregate_lengths)
    
    atol, rtol = 1e-3, 1e-5
    assert torch.allclose(output_logits, ref_output_logits, atol=atol, rtol=rtol)

