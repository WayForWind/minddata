

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RAND_AUGMENT_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RAND_AUGMENT_IR_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"
#include "OURSdata/dataset/kernels/image/rand_augment_op.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kRandAugmentOperation[] = "RandAugment";

class RandAugmentOperation : public TensorOperation {
 public:
  RandAugmentOperation(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins, InterpolationMode interpolation,
                       const std::vector<uint8_t> &fill_value);

  ~RandAugmentOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_ops_;
  int32_t magnitude_;
  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RAND_AUGMENT_IR_H_
