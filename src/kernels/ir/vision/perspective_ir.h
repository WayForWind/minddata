

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PERSPECTIVE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PERSPECTIVE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kPerspectiveOperation[] = "Perspective";

class PerspectiveOperation : public TensorOperation {
 public:
  PerspectiveOperation(const std::vector<std::vector<int32_t>> &start_points,
                       const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation,
                       const std::string &device_target = "CPU");

  ~PerspectiveOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPerspectiveOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  std::vector<std::vector<int32_t>> start_points_;
  std::vector<std::vector<int32_t>> end_points_;
  InterpolationMode interpolation_;
  std::string device_target_;  // CPU, Ascend
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PERSPECTIVE_IR_H_
