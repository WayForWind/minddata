

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_SLICE_PATCHES_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_SLICE_PATCHES_IR_H_

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
constexpr char kSlicePatchesOperation[] = "SlicePatches";

class SlicePatchesOperation : public TensorOperation {
 public:
  SlicePatchesOperation(int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value);

  ~SlicePatchesOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_height_;
  int32_t num_width_;
  SliceMode slice_mode_;
  uint8_t fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_SLICE_PATCHES_IR_H_
