/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PAD_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/device_tensor_ascend910b.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DvppPadOp : public TensorOp {
 public:
  DvppPadOp(int32_t pad_top, int32_t pad_bottom, int32_t pad_left, int32_t pad_right, BorderType padding_mode,
            uint8_t fill_r, uint8_t fill_g, uint8_t fill_b);

  ~DvppPadOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDvppCropOp; }

  bool IsDvppOp() override { return true; }

 private:
  int32_t pad_top_;
  int32_t pad_bottom_;
  int32_t pad_left_;
  int32_t pad_right_;
  BorderType boarder_type_;
  uint8_t fill_r_;
  uint8_t fill_g_;
  uint8_t fill_b_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PAD_OP_H_
