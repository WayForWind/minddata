

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_SELECT_SUBPOLICY_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_SELECT_SUBPOLICY_IR_H_

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
constexpr char kRandomSelectSubpolicyOperation[] = "RandomSelectSubpolicy";

class RandomSelectSubpolicyOperation : public TensorOperation {
 public:
  explicit RandomSelectSubpolicyOperation(
    const std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> &policy);

  ~RandomSelectSubpolicyOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_SELECT_SUBPOLICY_IR_H_
