

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ADJUST_SHARPNESS_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ADJUST_SHARPNESS_IR_H_

#include <map>
#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kAdjustSharpnessOperation[] = "AdjustSharpness";

class AdjustSharpnessOperation : public TensorOperation {
 public:
  explicit AdjustSharpnessOperation(float sharpness_factor, const std::string &device_target = "CPU");

  ~AdjustSharpnessOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kAdjustSharpnessOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  float sharpness_factor_;
  std::string device_target_;  // CPU, Ascend
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ADJUST_SHARPNESS_IR_H_
