

#include "OURSdata/dataset/kernels/data/duplicate_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status DuplicateOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1,
                               "Duplicate: only supports transform one column each time, got column num: " +
                                 std::to_string(input.size()) + ", check 'input_columns' when call this operator.");
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(input[0], &out));
  output->push_back(input[0]);
  output->push_back(out);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
