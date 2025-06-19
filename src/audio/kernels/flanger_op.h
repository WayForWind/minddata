
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FLANGER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FLANGER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class FlangerOp : public TensorOp {
 public:
  explicit FlangerOp(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
                     Modulation modulation, Interpolation interpolation)
      : sample_rate_(sample_rate),
        delay_(delay),
        depth_(depth),
        regen_(regen),
        width_(width),
        speed_(speed),
        phase_(phase),
        Modulation_(modulation),
        Interpolation_(interpolation) {}

  ~FlangerOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", delay:" << delay_ << ", depth: " << depth_
        << ", regen: " << regen_ << ", width: " << width_ << ", speed: " << speed_ << ", phase: " << phase_
        << ", Modulation: " << static_cast<int>(Modulation_) << ", Interpolation: " << static_cast<int>(Interpolation_)
        << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFlangerOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t sample_rate_;
  float delay_;
  float depth_;
  float regen_;
  float width_;
  float speed_;
  float phase_;
  Modulation Modulation_;
  Interpolation Interpolation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FLANGER_OP_H_
