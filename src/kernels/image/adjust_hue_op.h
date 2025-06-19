/

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_HUE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_HUE_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class AdjustHueOp : public TensorOp {
 public:
  explicit AdjustHueOp(float hue_factor) : hue_factor_(hue_factor) {}

  ~AdjustHueOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAdjustHueOp; }

 private:
  float hue_factor_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ADJUST_HUE_OP_H_
