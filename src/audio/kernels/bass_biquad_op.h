

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BASS_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BASS_BIQUAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class BassBiquadOp : public TensorOp {
 public:
  BassBiquadOp(int32_t sample_rate, float gain, float central_freq, float Q)
      : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}

  ~BassBiquadOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", gain: " << gain_ << ", central_freq: " << central_freq_
        << ", Q: " << Q_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kBassBiquadOp; }

 private:
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BASS_BIQUAD_OP_H_
