
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class ResizeOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;
  static const InterpolationMode kDefInterpolation;

  // Resizes the image to the output specified size. If only one value is provided,
  // the it will resize the smaller size and maintains the aspect ratio.
  // @param size1: the first size of output. If only this parameter is provided
  // the smaller dimension will be resized to this and then the other dimension changes
  // such that the aspect ratio is maintained.
  // @param size2: the second size of output. If this is also provided, the output size
  // will be (size1, size2)
  // @param InterpolationMode: the interpolation mode being used.
  explicit ResizeOp(int32_t size1, int32_t size2 = kDefWidth, InterpolationMode interpolation = kDefInterpolation)
      : size1_(size1), size2_(size2), interpolation_(interpolation) {}

  ResizeOp(const ResizeOp &rhs) = default;

  ResizeOp(ResizeOp &&rhs) = default;

  ~ResizeOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": " << size1_ << " " << size2_; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  static TensorShape ComputeOutputShape(const TensorShape &input, int32_t output_h, int32_t output_w);

  std::string Name() const override { return kResizeOp; }

 protected:
  int32_t size1_;
  int32_t size2_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_OP_H_
