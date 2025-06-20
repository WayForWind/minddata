

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASER_IR_H_

#include <map>
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
constexpr char kPhaserOperation[] = "Phaser";

class PhaserOperation : public TensorOperation {
 public:
  PhaserOperation(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
                  bool sinusoidal);

  ~PhaserOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPhaserOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  float gain_in_;
  float gain_out_;
  float delay_ms_;
  float decay_;
  float mod_speed_;
  bool sinusoidal_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PHASER_IR_H_
