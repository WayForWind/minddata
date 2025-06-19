/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_POSTERIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_POSTERIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DvppPosterizeOp : public TensorOp {
 public:
  explicit DvppPosterizeOp(uint8_t bits) : bits_(bits) {}

  ~DvppPosterizeOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDvppPosterizeOp; }

  bool IsDvppOp() override { return true; }

 private:
  uint8_t bits_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_HORIZONTAL_FLIP_OP_H_
