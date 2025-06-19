

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_IR_H_

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
constexpr char kNormalizeOperation[] = "Normalize";

class NormalizeOperation : public TensorOperation {
 public:
  NormalizeOperation(const std::vector<float> &mean, const std::vector<float> &std, bool is_hwc,
                     const std::string &device_target = "CPU");

  ~NormalizeOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  bool is_hwc_;
  std::string device_target_;  // CPU, Ascend
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_IR_H_
