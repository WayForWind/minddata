/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_AUTO_CONTRAST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_AUTO_CONTRAST_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppAutoContrastOp : public TensorOp {
 public:
  DvppAutoContrastOp(const std::vector<float> &cutoff, const std::vector<uint32_t> &ignore);

  ~DvppAutoContrastOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppAutoContrastOp; }

  bool IsDvppOp() override { return true; }

 private:
  std::vector<float> cutoff_;
  std::vector<uint32_t> ignore_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_AUTO_CONTRAST_OP_H_
