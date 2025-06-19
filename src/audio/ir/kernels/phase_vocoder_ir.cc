
#include "OURSdata/dataset/audio/ir/kernels/phase_vocoder_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/phase_vocoder_op.h"

namespace ours {
namespace dataset {
namespace audio {
PhaseVocoderOperation::PhaseVocoderOperation(float rate, const std::shared_ptr<Tensor> &phase_advance)
    : rate_(rate), phase_advance_(phase_advance) {}

PhaseVocoderOperation::~PhaseVocoderOperation() = default;

Status PhaseVocoderOperation::ValidateParams() {
  const int kPhaseAdvanceRank = 2;
  const int kLastDim = -1;
  const int kLastDimSize = 1;
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("PhaseVocoder", "rate", rate_));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    phase_advance_->Rank() == kPhaseAdvanceRank && phase_advance_->shape()[kLastDim] == kLastDimSize,
    "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.");
  return Status::OK();
}

std::string PhaseVocoderOperation::Name() const { return kPhaseVocoderOperation; }

std::shared_ptr<TensorOp> PhaseVocoderOperation::Build() {
  std::shared_ptr<PhaseVocoderOp> tensor_op = std::make_shared<PhaseVocoderOp>(rate_, phase_advance_);
  return tensor_op;
}

Status PhaseVocoderOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["rate"] = rate_;
  nlohmann::json phase_advance;
  RETURN_IF_NOT_OK(phase_advance_->to_json(&phase_advance));
  args["phase_advance"] = phase_advance;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
