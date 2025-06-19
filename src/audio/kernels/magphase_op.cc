
#include "OURSdata/dataset/audio/kernels/magphase_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
constexpr float MagphaseOp::kPower = 1.0;

Status MagphaseOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(ValidateTensorShape("Magphase", input[0]->IsComplex(), "<..., complex=2>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Magphase", input[0]));
  RETURN_IF_NOT_OK(Magphase(input, output, power_));
  return Status::OK();
}

Status MagphaseOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  auto vec = inputs[0].AsVector();
  vec.pop_back();
  auto out = TensorShape(vec);
  outputs = {out, out};
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "Magphase: invalid shape of input tensor.");
}

Status MagphaseOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Magphase", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
