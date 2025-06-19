/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_AFFINE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_AFFINE_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppAffineOp : public TensorOp {
 public:
  explicit DvppAffineOp(float degrees, const std::vector<float> &translation, float scale,
                        const std::vector<float> &shear, InterpolationMode interpolation,
                        const std::vector<uint8_t> &fill_value);

  ~DvppAffineOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDvppAffineOp; }

  bool IsDvppOp() override { return true; }

 private:
  float degrees_;
  std::vector<float> translation_;  // translation_x and translation_y
  float scale_;
  std::vector<float> shear_;  // shear_x and shear_y
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_AFFINE_OP_H_
