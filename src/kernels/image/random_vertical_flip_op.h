
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_VERTICAL_FLIP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_VERTICAL_FLIP_OP_H_

#include <memory>
#include <random>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomVerticalFlipOp : public RandomTensorOp {
 public:
  explicit RandomVerticalFlipOp(float prob) : distribution_(prob) {}

  ~RandomVerticalFlipOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomVerticalFlipOp; }

  uint32_t NumInput() override { return 1; }

  uint32_t NumOutput() override { return 1; }

 private:
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_VERTICAL_FLIP_OP_H_
