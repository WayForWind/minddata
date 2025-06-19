
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class SharpnessOp : public TensorOp {
 public:
  /// Default values, also used by bindings.cc
  static const float kDefAlpha;

  /// This class can be used to adjust the sharpness of an image.
  /// \@param[in] alpha A float indicating the enhancement factor.
  /// a factor of 0.0 gives a blurred image, a factor of 1.0 gives the
  /// original image, and a factor of 2.0 gives a sharpened image.

  explicit SharpnessOp(const float alpha = kDefAlpha) : alpha_(alpha) {}

  ~SharpnessOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSharpnessOp; }

 protected:
  float alpha_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SHARPNESS_OP_H_
