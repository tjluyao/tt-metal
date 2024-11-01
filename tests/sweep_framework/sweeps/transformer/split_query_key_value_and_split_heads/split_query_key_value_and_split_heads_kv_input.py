# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import (
    gen_shapes,
    sanitize_shape_rm,
    gen_rand_integers,
    gen_split_qkv_heads_spec,
    tensor_to_dtype,
)

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_spec": list(
            gen_split_qkv_heads_spec(
                batch_size_list=list(gen_rand_integers(1, 10, 2)),
                sequence_size_list=list(gen_rand_integers(1, 512, 4)),
                num_heads_list=list(gen_rand_integers(1, 20, 4)),
                transpose_key_list=[True, False],
                num_kv_heads_list=[None, 1],
                kv_input_tensor_list=[True],
            )
        ),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_dtype"] != test_vector["input_b_dtype"]:
        return True, "KV tensor dtype must be same as Q tensor dtype"
    if test_vector["input_spec"]["num_kv_heads"] != test_vector["input_spec"]["num_heads"]:
        return True, "Can't use num_kv_heads when using separate kv tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Inputs to eltwise binary must be tilized"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_a_dtype,
    input_b_dtype,
    input_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    batch_size, sequence_size, hidden_size, num_heads, num_kv_heads, _, transpose_key = input_spec.values()
    input_shape = (batch_size, sequence_size, hidden_size)

    if num_kv_heads == num_heads:
        num_kv_heads = num_heads

    head_size = hidden_size // (2 * num_kv_heads + num_heads)

    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.float32)
    intermediate_result = torch.reshape(
        torch_input_tensor, (batch_size, sequence_size, num_heads + (2 * num_kv_heads), head_size)
    )
    query, key, value = (
        intermediate_result[..., :num_heads, :],
        intermediate_result[..., num_heads : num_heads + num_kv_heads, :],
        intermediate_result[..., num_heads + num_kv_heads :, :],
    )

    query = tensor_to_dtype(torch.reshape(query, (batch_size, sequence_size, num_heads, head_size)), input_a_dtype)
    key = tensor_to_dtype(torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size)), input_b_dtype)
    value = tensor_to_dtype(torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size)), input_b_dtype)

    query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()
    key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
    value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()

    if transpose_key:
        key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

    torch_output_tensors = (query, key, value)

    q_hiden_size = head_size * num_heads

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor[:, :, :q_hiden_size],
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor[:, :, q_hiden_size:],
        dtype=input_b_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensors = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor_a,
        kv_input_tensor=input_tensor_b,
        num_heads=num_heads,
        num_kv_heads=None if num_kv_heads == num_heads else num_kv_heads,
        transpose_key=transpose_key,
        memory_config=output_memory_config,
    )
    e2e_perf = stop_measuring_time(start_time)

    passed = []
    output_string = ""
    for i in range(len(torch_output_tensors)):
        output_tensor = ttnn.to_torch(output_tensors[i])
        passed_, output_string_ = check_with_pcc(torch_output_tensors[i], output_tensor, 0.999)
        passed.append(passed_)
        output_string += output_string_ + ", "

    if all(passed):
        passed = True
    else:
        passed = False

    output_string = output_string[:-2]
    e2e_perf = stop_measuring_time(start_time)

    return [(passed, output_string), e2e_perf]