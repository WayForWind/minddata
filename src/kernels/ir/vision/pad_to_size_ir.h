

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kPadToSizeOperation[] = "PadToSize";

class PadToSizeOperation : public TensorOperation {
 public:
  PadToSizeOperation(const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                     const std::vector<uint8_t> &fill_value, BorderType padding_mode);

  ~PadToSizeOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> offset_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_
