/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppConvertColorOp : public TensorOp {
 public:
  explicit DvppConvertColorOp(ConvertMode convertMode) : convert_mode_(convertMode) {}

  ~DvppConvertColorOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppConvertColorOp; }

  bool IsDvppOp() override { return true; }

 private:
  ConvertMode convert_mode_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_
