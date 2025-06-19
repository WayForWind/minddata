
#include "OURSdata/dataset/audio/kernels/angle_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
Status AngleOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // if If the last dimension is not 2, then it's not a complex number
  RETURN_IF_NOT_OK(ValidateTensorShape("Angle", input->IsComplex(), "<..., complex=2>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Angle", input));
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    return Angle<double>(input, output);
  } else {
    std::shared_ptr<Tensor> tmp;
    TypeCast(input, &tmp, DataType(DataType::DE_FLOAT32));
    return Angle<float>(tmp, output);
  }
}

Status AngleOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  std::vector shape = inputs[0].AsVector();

  shape.pop_back();
  TensorShape out = TensorShape{shape};
  (void)outputs.emplace_back(out);
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "Angle: invalid shape of input tensor.");
}

Status AngleOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
