
#include "OURSdata/dataset/kernels/data/type_cast_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
TypeCastOp::TypeCastOp(const DataType &new_type) : type_(new_type) {}

TypeCastOp::TypeCastOp(const std::string &data_type) { type_ = DataType(data_type); }

Status TypeCastOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return TypeCast(input, output, type_);
}

Status TypeCastOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "TypeCast: inputs cannot be empty.");
  outputs[0] = type_;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
