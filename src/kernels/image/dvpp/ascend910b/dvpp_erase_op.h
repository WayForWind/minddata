/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_ERASE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_ERASE_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppEraseOp : public TensorOp {
 public:
  DvppEraseOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value)
      : top_(top), left_(left), height_(height), width_(width), value_(value) {}

  ~DvppEraseOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppEraseOp; }

  bool IsDvppOp() override { return true; }

 private:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<float> value_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_ERASE_OP_H_
