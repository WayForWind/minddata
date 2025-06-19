
#include "OURSdata/dataset/kernels/image/rescale_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RescaleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return Rescale(input, output, rescale_, shift_);
}

Status RescaleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Rescale: inputs cannot be empty.");
  outputs[0] = DataType(DataType::DE_FLOAT32);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
