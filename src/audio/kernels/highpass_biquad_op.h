

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_HIGHPASS_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_HIGHPASS_BIQUAD_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class HighpassBiquadOp : public TensorOp {
 public:
  HighpassBiquadOp(int32_t sample_rate, float cutoff_freq, float Q)
      : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}

  ~HighpassBiquadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kHighpassBiquadOp; };

 protected:
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_HIGHPASS_BIQUAD_OP_H_
