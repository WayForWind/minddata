

#include "OURSdata/dataset/audio/ir/kernels/mu_law_decoding_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/mu_law_decoding_op.h"

namespace ours {
namespace dataset {
namespace audio {
MuLawDecodingOperation::MuLawDecodingOperation(int32_t quantization_channels)
    : quantization_channels_(quantization_channels) {}

MuLawDecodingOperation::~MuLawDecodingOperation() = default;

Status MuLawDecodingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MuLawDecoding", "quantization_channels", quantization_channels_));
  return Status::OK();
}

Status MuLawDecodingOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["quantization_channels"] = quantization_channels_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> MuLawDecodingOperation::Build() {
  std::shared_ptr<MuLawDecodingOp> tensor_op = std::make_shared<MuLawDecodingOp>(quantization_channels_);
  return tensor_op;
}

std::string MuLawDecodingOperation::Name() const { return kMuLawDecodingOperation; }
}  // namespace audio
}  // namespace dataset
}  // namespace ours
