

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_AFFINE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_AFFINE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kAffineOperation[] = "Affine";

class AffineOperation : public TensorOperation {
 public:
  AffineOperation(float_t degrees, const std::vector<float> &translation, float scale, const std::vector<float> &shear,
                  InterpolationMode interpolation, const std::vector<uint8_t> &fill_value,
                  const std::string &device_target = "CPU");

  ~AffineOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  float degrees_;
  std::vector<float> translation_;
  float scale_;
  std::vector<float> shear_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
  std::string device_target_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_AFFINE_IR_H_
