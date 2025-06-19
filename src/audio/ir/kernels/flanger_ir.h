

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FLANGER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FLANGER_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {

constexpr char kFlangerOperation[] = "Flanger";

class FlangerOperation : public TensorOperation {
 public:
  explicit FlangerOperation(int32_t sample_rate, float delay, float depth, float regen, float width, float speed,
                            float phase, Modulation modulation, Interpolation interpolation);

  ~FlangerOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kFlangerOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float delay_;
  float depth_;
  float regen_;
  float width_;
  float speed_;
  float phase_;
  Modulation modulation_;
  Interpolation interpolation_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FLANGER_IR_H_
