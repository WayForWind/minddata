

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_HWC_TO_CHW_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_HWC_TO_CHW_IR_H_

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
constexpr char kHwcToChwOperation[] = "HwcToChw";

class HwcToChwOperation : public TensorOperation {
 public:
  HwcToChwOperation();

  ~HwcToChwOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_HWC_TO_CHW_IR_H_
