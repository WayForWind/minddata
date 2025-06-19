
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_

#include <memory>
#include <random>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/resize_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomResizeOp : public RandomTensorOp {
 public:
  RandomResizeOp(int32_t size_1, int32_t size_2) : size1_(size_1), size2_(size_2) {}

  ~RandomResizeOp() override = default;

  // Description: A function that prints info about the node
  void Print(std::ostream &out) const override { out << Name() << ": " << size1_ << " " << size2_; }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomResizeOp; }

  uint32_t NumInput() override { return 1; }

  uint32_t NumOutput() override { return 1; }

 private:
  int32_t size1_;
  int32_t size2_;
  std::uniform_int_distribution<int> distribution_{0, 3};
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
