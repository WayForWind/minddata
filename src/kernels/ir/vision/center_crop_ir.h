

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_CENTER_CROP_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_CENTER_CROP_IR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/include/dataset/vision.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kCenterCropOperation[] = "CenterCrop";

class CenterCropOperation : public TensorOperation {
 public:
  explicit CenterCropOperation(const std::vector<int32_t> &size);

  ~CenterCropOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<int32_t> size_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_CENTER_CROP_IR_H_
