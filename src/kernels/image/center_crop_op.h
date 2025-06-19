
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class CenterCropOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;

  explicit CenterCropOp(int32_t het, int32_t wid = kDefWidth) : crop_het_(het), crop_wid_(wid == 0 ? het : wid) {}

  ~CenterCropOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kCenterCropOp; }

 private:
  Status CenterCropImg(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) const;

  Status ConstructShape(const TensorShape &in_shape, std::shared_ptr<TensorShape> *out_shape) const;

  int32_t crop_het_;
  int32_t crop_wid_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_
