
#include "OURSdata/dataset/audio/kernels/amplitude_to_db_op.h"

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status AmplitudeToDBOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("AmplitudeToDB", input, kDefaultAudioDim, "<..., freq, time>"));

  std::shared_ptr<Tensor> input_tensor;

  float top_db = top_db_;
  float multiplier = stype_ == ScaleType::kPower ? 10.0 : 20.0;
  const float amin = 1e-10;
  float db_multiplier = std::log10(std::max(amin_, ref_value_));

  RETURN_IF_NOT_OK(ValidateTensorNumeric("AmplitudeToDB", input));
  // typecast
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return AmplitudeToDB<float>(input_tensor, output, multiplier, amin, db_multiplier, top_db);
  } else {
    input_tensor = input;
    return AmplitudeToDB<double>(input_tensor, output, multiplier, amin, db_multiplier, static_cast<double>(top_db));
  }
}
}  // namespace dataset
}  // namespace ours
