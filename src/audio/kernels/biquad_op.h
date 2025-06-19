
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BIQUAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class BiquadOp : public TensorOp {
 public:
  BiquadOp(float b0, float b1, float b2, float a0, float a1, float a2)
      : b0_(b0), b1_(b1), b2_(b2), a0_(a0), a1_(a1), a2_(a2) {}

  ~BiquadOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": b0: " << b0_ << ", b1: " << b1_ << ", b2: " << b2_ << ", a0: " << a0_ << ", a1: " << a1_
        << ", a2: " << a2_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kBiquadOp; }

 private:
  float b0_;
  float b1_;
  float b2_;
  float a0_;
  float a1_;
  float a2_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_BIQUAD_OP_H_
