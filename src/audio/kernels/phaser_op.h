

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class PhaserOp : public TensorOp {
 public:
  PhaserOp(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
           bool sinusoidal);

  ~PhaserOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kPhaserOp; };

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t sample_rate_;
  float gain_in_;
  float gain_out_;
  float delay_ms_;
  float decay_;
  float mod_speed_;
  bool sinusoidal_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASER_OP_H_
