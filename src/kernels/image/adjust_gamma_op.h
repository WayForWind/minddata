

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class AdjustGammaOp : public TensorOp {
 public:
  /// Default gain to be used
  static const float kGain;
  AdjustGammaOp(float gamma, float gain) : gamma_(gamma), gain_(gain) {}

  ~AdjustGammaOp() override = default;

  /// Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const AdjustGammaOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAdjustGammaOp; }

 private:
  float gamma_;
  float gain_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_
