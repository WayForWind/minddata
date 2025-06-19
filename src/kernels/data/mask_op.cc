

#include "OURSdata/dataset/kernels/data/mask_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status MaskOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::shared_ptr<Tensor> temp_output;
  CHECK_FAIL_RETURN_UNEXPECTED(type_.IsNumeric(), "Mask: only support numeric datatype of input, got string.");

  RETURN_IF_NOT_OK(Mask(input, &temp_output, value_, op_));

  // cast the output to the the required type. Skip casting if type_ is bool.
  if (type_ != DataType::DE_BOOL) {
    RETURN_IF_NOT_OK(cast_->Compute(temp_output, output));
  } else {
    *output = std::move(temp_output);
  }

  return Status::OK();
}

Status MaskOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "Mask: inputs cannot be empty.");
  outputs[0] = type_;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
