

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_

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
constexpr char kRotateOperation[] = "Rotate";

class RotateOperation : public TensorOperation {
 public:
  explicit RotateOperation(FixRotationAngle angle);

  RotateOperation(float degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
                  const std::vector<uint8_t> &fill_value, const std::string &device_target = "CPU");

  ~RotateOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  void setAngle(uint64_t angle_id);

  MapTargetDevice Type() override;

 private:
  uint64_t angle_id_;
  float degrees_;
  InterpolationMode interpolation_mode_;
  bool expand_;
  std::vector<float> center_;
  std::vector<uint8_t> fill_value_;
  std::shared_ptr<TensorOp> rotate_op_;
  std::string device_target_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_
