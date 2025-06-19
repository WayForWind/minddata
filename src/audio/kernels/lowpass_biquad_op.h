

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_LOWPASS_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_LOWPASS_BIQUAD_OP_H_

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class LowpassBiquadOp : public TensorOp {
 public:
  /// default values;
  static const float kQ;

  LowpassBiquadOp(int32_t sample_rate, float cutoff_freq, float Q)
      : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}

  ~LowpassBiquadOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", cutoff_freq: " << cutoff_freq_ << ", Q: " << Q_
        << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kLowpassBiquadOp; }

 private:
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_LOWPASS_BIQUAD_OP_H_
