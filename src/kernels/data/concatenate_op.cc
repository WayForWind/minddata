

#include "OURSdata/dataset/kernels/data/concatenate_op.h"

#include <limits>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status ConcatenateOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(Concatenate(input, output, axis_, prepend_, append_));
  return Status::OK();
}

Status ConcatenateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));

  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Concatenate: inputs can not be empty.");

  std::vector<TensorShape> inputs_copy;
  inputs_copy.push_back(inputs[0].Squeeze());

  CHECK_FAIL_RETURN_UNEXPECTED(inputs.at(0).Rank() == 1,
                               "Concatenate: only 1D input supported, got rank:" + std::to_string(inputs.at(0).Rank()));

  outputs.clear();
  dsize_t output_shape = 0;
  output_shape = output_shape + inputs.at(0).NumOfElements();
  if (prepend_ != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(prepend_->shape().Rank() == 1, "Concatenate: only 1D prepend supported, got rank: " +
                                                                  std::to_string(prepend_->shape().Rank()));
    CHECK_FAIL_RETURN_UNEXPECTED(
      (std::numeric_limits<uint64_t>::max() - output_shape) > prepend_->shape().NumOfElements(),
      "Concatenate: append parameter is too large to pend.");
    output_shape = output_shape + prepend_->shape().NumOfElements();
  }
  if (append_ != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(append_->shape().Rank() == 1, "Concatenate: only 1D append supported, got rank: " +
                                                                 std::to_string(append_->shape().Rank()));
    CHECK_FAIL_RETURN_UNEXPECTED(
      (std::numeric_limits<uint64_t>::max() - output_shape) > append_->shape().NumOfElements(),
      "Concatenate: append parameter is too large to pend, got: " + std::to_string(append_->shape().NumOfElements()));
    output_shape = output_shape + append_->shape().NumOfElements();
  }

  (void)outputs.emplace_back(std::vector<dsize_t>{output_shape});
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
