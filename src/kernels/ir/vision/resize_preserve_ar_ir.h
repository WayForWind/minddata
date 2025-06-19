

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RESIZE_PRESERVE_AR_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RESIZE_PRESERVE_AR_IR_H_

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
constexpr char kResizePreserveAROperation[] = "ResizePreserveAR";

class ResizePreserveAROperation : public TensorOperation {
 public:
  ResizePreserveAROperation(int32_t height, int32_t width, int32_t img_orientation);

  ~ResizePreserveAROperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t height_;
  int32_t width_;
  int32_t img_orientation_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RESIZE_PRESERVE_AR_IR_H_
