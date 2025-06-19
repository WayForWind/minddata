/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_EQUALIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_EQUALIZE_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppEqualizeOp : public TensorOp {
 public:
  DvppEqualizeOp() {}

  ~DvppEqualizeOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppEqualizeOp; }

  bool IsDvppOp() override { return true; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_EQUALIZE_OP_H_
