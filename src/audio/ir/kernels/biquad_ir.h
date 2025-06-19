
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BIQUAD_IR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kBiquadOperation[] = "Biquad";

class BiquadOperation : public TensorOperation {
 public:
  BiquadOperation(float b0, float b1, float b2, float a0, float a1, float a2);

  ~BiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float b0_;
  float b1_;
  float b2_;
  float a0_;
  float a1_;
  float a2_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_BIQUAD_IR_H_
