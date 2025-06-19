

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_

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
constexpr char kRandomAffineOperation[] = "RandomAffine";

class RandomAffineOperation : public TensorOperation {
 public:
  RandomAffineOperation(const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                        const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                        InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

  ~RandomAffineOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<float_t> degrees_;          // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_AFFINE_IR_H_
