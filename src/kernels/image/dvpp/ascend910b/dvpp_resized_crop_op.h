/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_RESIZED_CROP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_RESIZED_CROP_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppResizedCropOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;
  static const InterpolationMode kDefInterpolation;

  DvppResizedCropOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
                    InterpolationMode interpolation)
      : top_(top), left_(left), height_(height), width_(width), size_(size), interpolation_(interpolation) {}

  ~DvppResizedCropOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  TensorShape ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w);

  std::string Name() const override { return kDvppResizedCropOp; }

  bool IsDvppOp() override { return true; }

 private:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_RESIZED_CROP_OP_H_
