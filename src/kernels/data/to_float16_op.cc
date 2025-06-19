
#include "OURSdata/dataset/kernels/data/to_float16_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status ToFloat16Op::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return ToFloat16(input, output);
}

Status ToFloat16Op::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "ToFloat16: inputs cannot be empty.");
  outputs[0] = DataType(DataType::DE_FLOAT16);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
