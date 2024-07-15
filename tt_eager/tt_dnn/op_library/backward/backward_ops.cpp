// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/backward/backward_ops.hpp"

#include "tt_dnn/op_library/complex/complex_ops.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/cpp/ttnn/operations/embedding/embedding/embedding.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace tt {

namespace tt_metal {

// Autoformat support
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config) {
    auto formatted_input_tensor = temp;
    if(formatted_input_tensor.get_layout()==Layout::ROW_MAJOR){
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_legacy_shape(), false, false, true, true);
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return formatted_input_tensor;
}

// Prod
// along a single dimension --> result: grad_data * (y / input )
std::vector<Tensor> _prod_bw(
    const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor prod_result = prod(input, all_dimensions, dim, output_mem_config);
    if(prod_result.get_layout()==Layout::ROW_MAJOR && prod_result.storage_type() == StorageType::DEVICE){
        prod_result = tt::tt_metal::change_layout_to_tile(prod_result, output_mem_config);
        }
    if (all_dimensions == true) {
        Tensor temp =
            ttnn::multiply(prod_result, grad, std::nullopt, output_mem_config);  // result is stored in the first position
        Tensor fill_tensor = tt::numpy::fill_first_val_into_tensor<bfloat16>(
            temp, temp.get_dtype(), temp.get_layout(), temp.device(), output_mem_config);
        Tensor all_dimension_result =
            ttnn::multiply(ttnn::reciprocal(input, output_mem_config), fill_tensor, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(all_dimension_result);
        return grad_tensor;
    }
    // all_dimensions = False
    Tensor updated_grad = prod_result;
    if (prod_result.get_legacy_shape() != grad.get_legacy_shape()) {
        if (dim == 3 || dim == -1) {
            std::vector<int64_t> after_permute_dims = {0, 3, 1, 2};
            Tensor required = permute(grad, after_permute_dims, output_mem_config);
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
                grad.get_legacy_shape()[0] - 1, 0, grad.get_legacy_shape()[1] - 1, grad.get_legacy_shape()[2] - 1};
            Tensor new_slice_tensor = ttnn::slice(0, required, start_index, end_index, std::nullopt);
            after_permute_dims = {0, 2, 3, 1};
            updated_grad = permute(new_slice_tensor, after_permute_dims, output_mem_config);
            Tensor pad_updated_grad = updated_grad.pad_to_tile(1.0f);
            Tensor pad_prod_result = prod_result.pad_to_tile(1.0f);
            pad_updated_grad = pad_updated_grad.to(Layout::TILE);
            pad_prod_result = pad_prod_result.to(Layout::TILE);
            updated_grad = pad_updated_grad.to(input.device());
            prod_result = pad_prod_result.to(input.device());
            pad_updated_grad.deallocate();
            pad_prod_result.deallocate();
        } else if (dim == 2 || dim == -2) {
            std::vector<int64_t> after_permute_dims = {0, 2, 1, 3};
            Tensor required = permute(grad, after_permute_dims, output_mem_config);
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
                grad.get_legacy_shape()[0] - 1, 0, grad.get_legacy_shape()[1] - 1, grad.get_legacy_shape()[3] - 1};
            Tensor new_slice_tensor = ttnn::slice(0, required, start_index, end_index, std::nullopt);
            updated_grad = permute(new_slice_tensor, after_permute_dims, output_mem_config);
            if(updated_grad.get_layout()==Layout::ROW_MAJOR){
                updated_grad = tt::tt_metal::change_layout_to_tile(updated_grad, output_mem_config);
            }
        }
    }
    Tensor reciprocal_input = ttnn::reciprocal(input, output_mem_config);
    Tensor temp = ttnn::multiply(prod_result, (dim == 1 || dim == 0 || dim == -4 || dim == -3) ? grad : updated_grad, std::nullopt, output_mem_config);
    if(temp.get_layout()==Layout::ROW_MAJOR){
        temp = tt::tt_metal::change_layout_to_tile(temp, output_mem_config);
    }
    if (dim == 3 || dim == -1) {
        Tensor grad_result = bcast(reciprocal_input, temp, BcastOpMath::MUL, BcastOpDim::W, output_mem_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 2 || dim == -2) {
        Tensor grad_result = bcast(reciprocal_input, temp, BcastOpMath::MUL, BcastOpDim::H, output_mem_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    } else if (dim == 1 || dim == -3) {
        Tensor tensor_1_temp = reciprocal_input;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, 0},
                          {0, 32 - (reciprocal_input.get_legacy_shape()[1] % 32)},
                          {0, 0},
                          {0, 0}};
            tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, true, std::nullopt);
        }
        std::vector<int64_t> after_permute_dims = {0, 2, 3, 1};
        Tensor tensor_1 = permute(tensor_1_temp, after_permute_dims, output_mem_config);
        Tensor tensor_2 = permute(temp, after_permute_dims, output_mem_config);

        // put the tensor back on device because permute throws it off device
        // See: Remove auto format within permute_op.cpp #9404
        tensor_2 = AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

        after_permute_dims = {0, 3, 1, 2};
        Tensor result = permute(
            bcast(tensor_1, tensor_2, BcastOpMath::MUL, BcastOpDim::W, output_mem_config),
            after_permute_dims,
            output_mem_config);
        Tensor grad_result = result;
        if (reciprocal_input.get_legacy_shape()[1] % 32 != 0) {
            std::vector<uint32_t> start_index = {0, 0, 0, 0};
            std::vector<uint32_t> end_index = {
                input.get_legacy_shape()[0] - 1,
                input.get_legacy_shape()[1] - 1,
                input.get_legacy_shape()[2] - 1,
                input.get_legacy_shape()[3] - 1};
            grad_result = ttnn::slice(0, result, start_index, end_index, std::nullopt);
        }
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    // dim 0
    Tensor tensor_1_temp = reciprocal_input;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        std::vector<std::pair<uint32_t, uint32_t>> padding = {{0, (32 - (reciprocal_input.get_legacy_shape()[0] % 32))},
                      {0, 0},
                      {0, 0},
                      {0, 0}};
        tensor_1_temp = ttnn::pad(0, reciprocal_input, padding, 0, false, std::nullopt);
    }
    std::vector<int64_t> after_permute_dims = {3, 1, 2, 0};
    Tensor tensor_1 = permute(tensor_1_temp, after_permute_dims, output_mem_config);
    Tensor tensor_2 = permute(temp, after_permute_dims, output_mem_config);

    // put the tensor back on device because permute throws it off device
    // See: Remove auto format within permute_op.cpp #9404
    tensor_2 = AutoFormat::move_tensor_to_device_and_pad(tensor_2, tensor_1.device(),tensor_1.get_layout(), tensor_1.memory_config());

    Tensor result = permute(
        bcast(tensor_1, tensor_2, BcastOpMath::MUL, BcastOpDim::W, output_mem_config),
        after_permute_dims,
        output_mem_config);
    Tensor grad_result = result;
    if (reciprocal_input.get_legacy_shape()[0] % 32 != 0) {
        std::vector<uint32_t> start_index = {0, 0, 0, 0};
        std::vector<uint32_t> end_index = {
            input.get_legacy_shape()[0] - 1,
            input.get_legacy_shape()[1] - 1,
            input.get_legacy_shape()[2] - 1,
            input.get_legacy_shape()[3] - 1};
        grad_result = ttnn::slice(0, result, start_index, end_index, std::nullopt);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> prod_bw(
    const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _prod_bw)(grad, input, all_dimensions, dim, output_mem_config);
}

#define CHECK_FOR_COMPLEX(input)                                                     \
    do {                                                                             \
        TT_ASSERT(utility::is_complex_shape(input), "works for complex shape only"); \
        /* TT_ASSERT( input.shape()[0] == 1, "tensor should have batch size 1"); */  \
    } while (0);

// complex conj
// self: grad.conj()
std::vector<Tensor> _conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> conj_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _conj_bw)(grad, input, output_mem_config);
}

// complex reciprocal
// self: -grad * (result * result).conj()
std::vector<Tensor> _complex_recip_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor input_r = real(input, output_mem_config);
    Tensor input_i = imag(input, output_mem_config);
    Tensor condition_nan =
        ttnn::logical_and(ttnn::eqz(input_r, output_mem_config), ttnn::eqz(input_i, output_mem_config), std::nullopt, output_mem_config);
    input_r.deallocate();
    input_i.deallocate();
    Tensor nan_flag = mk_complex(condition_nan, condition_nan, output_mem_config);
    condition_nan.deallocate();
    Tensor grad_result = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_mul(
            ttnn::neg(grad, output_mem_config),
            conj(
                complex_mul(
                    complex_recip(input, output_mem_config),
                    complex_recip(input, output_mem_config),
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    nan_flag.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> complex_recip_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_recip_bw)(grad, input, output_mem_config);
}

// complex imag
// imag: at::imag(grad)
std::vector<Tensor> _imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        mk_complex(zeros_like(real(input, output_mem_config), output_mem_config), grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> imag_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _imag_bw)(grad, input, output_mem_config);
}

// complex real
// real: at::real(grad)
std::vector<Tensor> _real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor grad_result =
        mk_complex(grad, zeros_like(imag(input, output_mem_config), output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> real_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _real_bw)(grad, input, output_mem_config);
}

// angle at::where(self == 0.0, at::zeros({}, self.options()), grad * self / self.abs().pow(2)
std::vector<Tensor> _angle_bw(
    const Tensor& grad, const Tensor& input, bool is_complextensor, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    if (is_complextensor) {
        CHECK_FOR_COMPLEX(input);
        Tensor inp_r = real(input, output_mem_config);
        Tensor inp_i = imag(input, output_mem_config);
        Tensor condition_zero =
            ttnn::logical_and(ttnn::eqz(inp_r, output_mem_config), ttnn::eqz(inp_i, output_mem_config), std::nullopt, output_mem_config);
        Tensor abs_squared = ttnn::reciprocal(
            ttnn::add(ttnn::square(inp_r, output_mem_config), ttnn::square(inp_i, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);
        Tensor real = where(
            condition_zero,
            zeros_like(inp_r, output_mem_config),
            ttnn::multiply(grad,
                ttnn::multiply(ttnn::neg(inp_i, output_mem_config), abs_squared, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            output_mem_config);
        Tensor imag = where(
            condition_zero,
            zeros_like(inp_i, output_mem_config),
            ttnn::multiply(grad, ttnn::multiply(inp_r, abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);
        condition_zero.deallocate();
        abs_squared.deallocate();
        inp_r.deallocate();
        inp_i.deallocate();
        Tensor grad_result = mk_complex(real, imag, output_mem_config);
        real.deallocate();
        imag.deallocate();
        grad_tensor.emplace_back(grad_result);
    } else {
        Tensor grad_result = zeros_like(grad, output_mem_config);
        grad_tensor.emplace_back(grad_result);
    }
    return grad_tensor;
}
std::vector<Tensor> angle_bw(
    const Tensor& grad, const Tensor& input, bool is_complextensor, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _angle_bw)(grad, input, is_complextensor, output_mem_config);
}

// complex abs
// self: grad * self.sgn()
std::vector<Tensor> _complex_abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> grad_tensor;
    Tensor result = complex_abs(input, output_mem_config);
    result = mk_complex(result, result, output_mem_config);
    Tensor grad_c = mk_complex(grad, grad, output_mem_config);
    Tensor grad_result = where(
        ttnn::eqz(result, output_mem_config),
        zeros_like(result, output_mem_config),
        ttnn::multiply(grad_c,
            ttnn::multiply(input, ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> complex_abs_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_abs_bw)(grad, input, output_mem_config);
}
// polar
// grad_abs = torch.real(grad_conj * torch.sgn(result))
// result_mul_1_j = result * torch.tensor(0.0 + 1.0j)
// grad_angle = torch.real(grad_conj * result_mul_1_j)
// polar fwd op uses sin and cos hence input_b range is (0, 2*pi)
std::vector<Tensor> _polar_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor result = polar(input_a, input_b, output_mem_config);
    Tensor abs_result = complex_abs(result, output_mem_config);
    abs_result = mk_complex(abs_result, abs_result, output_mem_config);
    Tensor sgn_result = where(
        ttnn::eqz(abs_result, output_mem_config),
        zeros_like(result, output_mem_config),
        ttnn::multiply(result, ttnn::reciprocal(abs_result, output_mem_config), std::nullopt, output_mem_config),
        output_mem_config);
    abs_result.deallocate();
    Tensor grad_abs =
        real(complex_mul(conj(grad, output_mem_config), sgn_result, output_mem_config), output_mem_config);
    sgn_result.deallocate();
    Tensor flip_tensor = mk_complex(
        zeros_like(input_a, output_mem_config), full_like(input_b, 1.0, output_mem_config), output_mem_config);
    Tensor grad_angle = real(
        complex_mul(
            conj(grad, output_mem_config), complex_mul(result, flip_tensor, output_mem_config), output_mem_config),
        output_mem_config);
    result.deallocate();
    flip_tensor.deallocate();
    Tensor grad_result = mk_complex(grad_abs, grad_angle, output_mem_config);
    grad_abs.deallocate();
    grad_angle.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
std::vector<Tensor> polar_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polar_bw)(grad, input_a, input_b, output_mem_config);
}

// complex div
//  self: grad / other.conj();
//  other: -grad * ((self / other) / other).conj();
std::vector<Tensor> _complex_div_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor other_r = real(other, output_mem_config);
    Tensor other_i = imag(other, output_mem_config);
    Tensor condition_nan =
        ttnn::logical_and(ttnn::eqz(other_r, output_mem_config), ttnn::eqz(other_i, output_mem_config), std::nullopt, output_mem_config);
    other_r.deallocate();
    other_i.deallocate();
    Tensor nan_flag = mk_complex(condition_nan, condition_nan, output_mem_config);
    condition_nan.deallocate();
    Tensor grad_a = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_div(grad, conj(other, output_mem_config), output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor result = complex_div(input, other, output_mem_config);
    Tensor grad_b = where(
        nan_flag,
        full_like(input, std::nanf(""), output_mem_config),
        complex_mul(
            ttnn::neg(grad, output_mem_config),
            conj(complex_div(result, other, output_mem_config), output_mem_config),
            output_mem_config),
        output_mem_config);
    result.deallocate();
    nan_flag.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_div_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_div_bw)(grad, input, other, output_mem_config);
}

// complex mul
// grad_input = grad * other.conj()
// grad_other = grad * input.conj()
std::vector<Tensor> _complex_mul_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = complex_mul(grad, conj(other, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = complex_mul(grad, conj(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_mul_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_mul_bw)(grad, input, other, output_mem_config);
}

// complex add
// self: grad, other: grad * alpha
std::vector<Tensor> _complex_add_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_b = ttnn::multiply(grad, alpha, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_add_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_add_bw)(grad, input, other, alpha, output_mem_config);
}

// complex sub
// self: grad, other: -grad * alpha
std::vector<Tensor> _complex_sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    CHECK_FOR_COMPLEX(other);
    CHECK_FOR_COMPLEX(grad);
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::NEG},
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha}};
    Tensor grad_b = ttnn::unary_chain(grad, ops_chain, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}
std::vector<Tensor> complex_sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _complex_sub_bw)(grad, input, other, alpha, output_mem_config);
}
#undef CHECK_FOR_COMPLEX

}  // namespace tt_metal

}  // namespace tt
