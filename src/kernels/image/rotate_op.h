

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ROTATE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ROTATE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RotateOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const std::vector<float> kDefCenter;
  static const InterpolationMode kDefInterpolation;
  static const bool kDefExpand;
  static const uint8_t kDefFillR;
  static const uint8_t kDefFillG;
  static const uint8_t kDefFillB;

  /// Constructor
  explicit RotateOp(int angle_id);

  explicit RotateOp(float degrees, InterpolationMode resample = kDefInterpolation, bool expand = kDefExpand,
                    std::vector<float> center = kDefCenter, uint8_t fill_r = kDefFillR, uint8_t fill_g = kDefFillG,
                    uint8_t fill_b = kDefFillB);

  ~RotateOp() override = default;

  TensorShape ConstructShape(const TensorShape &in_shape) const;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kRotateOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  void setAngle(uint64_t angle_id) { angle_id_ = angle_id; }

  /// Member variables
 protected:
  uint64_t angle_id_;

 private:
  float degrees_;
  std::vector<float> center_;
  InterpolationMode interpolation_;
  bool expand_;
  uint8_t fill_r_;
  uint8_t fill_g_;
  uint8_t fill_b_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ROTATE_OP_H_
