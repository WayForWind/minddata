/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_GAUSSIAN_BLUR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_GAUSSIAN_BLUR_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppGaussianBlurOp : public TensorOp {
 public:
  DvppGaussianBlurOp(int32_t kernel_x, int32_t kernel_y, float sigma_x, float sigma_y)
      : kernel_x_(kernel_x), kernel_y_(kernel_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {}

  ~DvppGaussianBlurOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppGaussianBlurOp; }

  bool IsDvppOp() override { return true; }

 private:
  int32_t kernel_x_;
  int32_t kernel_y_;
  float sigma_x_;
  float sigma_y_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_GAUSSIAN_BLUR_OP_H_
