
#include "OURSdata/dataset/audio/kernels/complex_norm_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"

namespace ours {
namespace dataset {
// constructor
ComplexNormOp::ComplexNormOp(float power) : power_(power) {}

// main function
Status ComplexNormOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateTensorShape("ComplexNorm", input->IsComplex(), "<..., complex=2>"));
  return ComplexNorm(input, output, power_);
}

Status ComplexNormOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto input_size = inputs[0].AsVector();
  input_size.pop_back();
  TensorShape out = TensorShape(input_size);
  outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "ComplexNorm: invalid shape of input tensor.");
}

Status ComplexNormOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(
    ValidateTensorType("ComplexNorm", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}

void ComplexNormOp::Print(std::ostream &out) const { out << "ComplexNormOp: power " << power_; }
}  // namespace dataset
}  // namespace ours
