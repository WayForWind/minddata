

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_POSTERIZE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_POSTERIZE_IR_H_

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
constexpr char kPosterizeOperation[] = "Posterize";

class PosterizeOperation : public TensorOperation {
 public:
  explicit PosterizeOperation(uint8_t bits, const std::string &device_target = "CPU");

  ~PosterizeOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPosterizeOperation; };

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  MapTargetDevice Type() override;

 private:
  uint8_t bits_;
  std::string device_target_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_POSTERIZE_IR_H_
