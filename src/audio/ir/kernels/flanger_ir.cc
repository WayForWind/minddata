

#include "OURSdata/dataset/audio/ir/kernels/flanger_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/flanger_op.h"

namespace ours {
namespace dataset {
namespace audio {
// FlangerOperation
FlangerOperation::FlangerOperation(int32_t sample_rate, float delay, float depth, float regen, float width, float speed,
                                   float phase, Modulation modulation, Interpolation interpolation)
    : sample_rate_(sample_rate),
      delay_(delay),
      depth_(depth),
      regen_(regen),
      width_(width),
      speed_(speed),
      phase_(phase),
      modulation_(modulation),
      interpolation_(interpolation) {}

Status FlangerOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("Flanger", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "delay", delay_, {0, 30}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "depth", depth_, {0, 10}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "regen", regen_, {-95, 95}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "width", width_, {0, 100}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "speed", speed_, {0.1, 10}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Flanger", "phase", phase_, {0, 100}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> FlangerOperation::Build() {
  std::shared_ptr<FlangerOp> tensor_op = std::make_shared<FlangerOp>(sample_rate_, delay_, depth_, regen_, width_,
                                                                     speed_, phase_, modulation_, interpolation_);
  return tensor_op;
}

Status FlangerOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["delay"] = delay_;
  args["depth"] = depth_;
  args["regen"] = regen_;
  args["width"] = width_;
  args["speed"] = speed_;
  args["phase"] = phase_;
  args["modulation"] = modulation_;
  args["interpolation"] = interpolation_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
