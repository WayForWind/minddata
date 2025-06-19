

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_BOUNDING_BOX_AUGMENT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_BOUNDING_BOX_AUGMENT_OP_H_

#include <cstdlib>
#include <memory>
#include <random>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class BoundingBoxAugmentOp : public RandomTensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const float kDefRatio;

  // Constructor for BoundingBoxAugmentOp
  // @param std::shared_ptr<TensorOp> transform transform: C++ operation to apply on select bounding boxes
  // @param float ratio: ratio of bounding boxes to have the transform applied on
  BoundingBoxAugmentOp(std::shared_ptr<TensorOp> transform, float ratio);

  ~BoundingBoxAugmentOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const BoundingBoxAugmentOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kBoundingBoxAugmentOp; }

 private:
  float ratio_;
  std::uniform_real_distribution<float> uniform_;
  std::shared_ptr<TensorOp> transform_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_BOUNDING_BOX_AUGMENT_OP_H_
