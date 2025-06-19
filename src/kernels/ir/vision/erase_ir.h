

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ERASE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ERASE_IR_H_

#include <algorithm>
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
constexpr char kEraseOperation[] = "Erase";

class EraseOperation : public TensorOperation {
 public:
  EraseOperation(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value,
                 bool inplace, const std::string &device_target = "CPU");

  ~EraseOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<float> value_;
  bool inplace_;
  std::string device_target_;  // CPU, Ascend
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_ERASE_IR_H_
