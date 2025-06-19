

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_

#include <cmath>
#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class EqualizerBiquadOp : public TensorOp {
 public:
  static const float kQ;

  EqualizerBiquadOp(int32_t sample_rate, float center_freq, float gain, float Q)
      : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}

  ~EqualizerBiquadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kEqualizerBiquadOp; }

 protected:
  int32_t sample_rate_;
  float center_freq_;
  float gain_;
  float Q_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_
