
#include "OURSdata/dataset/audio/ir/kernels/mu_law_encoding_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/mu_law_encoding_op.h"

namespace ours {
namespace dataset {
namespace audio {
MuLawEncodingOperation::MuLawEncodingOperation(int32_t quantization_channels)
    : quantization_channels_(quantization_channels) {}

MuLawEncodingOperation::~MuLawEncodingOperation() = default;

Status MuLawEncodingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MuLawEncoding", "quantization_channels", quantization_channels_));
  return Status::OK();
}

Status MuLawEncodingOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["quantization_channels"] = quantization_channels_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> MuLawEncodingOperation::Build() {
  std::shared_ptr<MuLawEncodingOp> tensor_op = std::make_shared<MuLawEncodingOp>(quantization_channels_);
  return tensor_op;
}

std::string MuLawEncodingOperation::Name() const { return kMuLawEncodingOperation; }
}  // namespace audio
}  // namespace dataset
}  // namespace ours
