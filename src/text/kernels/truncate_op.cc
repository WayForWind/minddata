/
#include "OURSdata/dataset/text/kernels/truncate_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/slice_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/data_utils.h"

namespace ours {
namespace dataset {
Status TruncateOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  constexpr int kMaxSeqRank = 2;
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Rank() == 1 || input->shape().Rank() == kMaxSeqRank,
                               "Truncate: the input tensor should be of dimension 1 or 2.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type() == DataType::DE_STRING || input->type().IsNumeric(),
    "Truncate: Truncate: the input tensor should be in type of [bool, int, float, double, string].");
  return Truncate(input, output, max_seq_len_);
}

Status TruncateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  constexpr int kMaxSeqRank = 2;
  CHECK_FAIL_RETURN_UNEXPECTED(inputs[0].Rank() == 1 || inputs[0].Rank() == kMaxSeqRank,
                               "Truncate: the input tensor should be of dimension 1 or 2.");
  if (inputs[0].Rank() == 1) {
    outputs.clear();
    auto shape = inputs[0].AsVector();
    int length = shape[0];
    shape[0] = std::min(length, max_seq_len_);
    (void)outputs.emplace_back(TensorShape{shape});
  } else {
    outputs.clear();
    auto shape = inputs[0].AsVector();
    int length = shape[1];
    shape[1] = std::min(length, max_seq_len_);
    (void)outputs.emplace_back(TensorShape{shape});
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
