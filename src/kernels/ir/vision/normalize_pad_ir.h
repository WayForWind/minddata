

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_PAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_PAD_IR_H_

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
constexpr char kNormalizePadOperation[] = "NormalizePad";

class NormalizePadOperation : public TensorOperation {
 public:
  NormalizePadOperation(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype,
                        bool is_hwc);

  ~NormalizePadOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
  bool is_hwc_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_NORMALIZE_PAD_IR_H_
