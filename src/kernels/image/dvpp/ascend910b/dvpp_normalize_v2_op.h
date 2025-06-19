
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_NORMALIZE_V2_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_NORMALIZE_V2_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DvppNormalizeV2Op : public TensorOp {
 public:
  DvppNormalizeV2Op(std::vector<float> mean, std::vector<float> std, bool is_hwc);

  ~DvppNormalizeV2Op() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppNormalizeOp; }

  bool IsDvppOp() override { return true; }

  bool IsHWC() override { return is_hwc_; }

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  bool is_hwc_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_NORMALIZE_V2_OP_H_
