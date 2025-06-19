
#include "OURSdata/dataset/kernels/ir/vision/to_tensor_ir.h"

#include "OURSdata/dataset/kernels/image/to_tensor_op.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
ToTensorOperation::ToTensorOperation(const std::string &output_type) {
  DataType temp_output_type(output_type);
  output_type_ = temp_output_type;
}

ToTensorOperation::ToTensorOperation(const DataType &output_type) { output_type_ = output_type; }

ToTensorOperation::~ToTensorOperation() = default;

std::string ToTensorOperation::Name() const { return kToTensorOperation; }

Status ToTensorOperation::ValidateParams() {
  if (output_type_ == DataType::DE_UNKNOWN) {
    std::string err_msg = "ToTensor: Invalid data type for output_type parameter.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ToTensorOperation::Build() { return std::make_shared<ToTensorOp>(output_type_); }

Status ToTensorOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["output_type"] = output_type_.ToString();
  *out_json = args;
  return Status::OK();
}

Status ToTensorOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "output_type", kToTensorOperation));
  std::string output_type = op_params["output_type"];
  *operation = std::make_shared<vision::ToTensorOperation>(output_type);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
