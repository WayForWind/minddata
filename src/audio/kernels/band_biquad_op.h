
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BAND_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BAND_BIQUAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class BandBiquadOp : public TensorOp {
 public:
  BandBiquadOp(int32_t sample_rate, float central_freq, float Q, bool noise)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), noise_(noise) {}

  ~BandBiquadOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", central_freq: " << central_freq_ << ", Q: " << Q_
        << ", noise: " << noise_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kBandBiquadOp; }

 private:
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool noise_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BAND_BIQUAD_OP_H_
