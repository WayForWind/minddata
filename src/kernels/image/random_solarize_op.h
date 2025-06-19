

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SOLARIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SOLARIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomSolarizeOp : public RandomTensorOp {
 public:
  // Pick a random threshold value to solarize the image with
  explicit RandomSolarizeOp(const std::vector<uint8_t> &threshold) : threshold_(threshold) {}

  ~RandomSolarizeOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomSolarizeOp; }

 private:
  std::vector<uint8_t> threshold_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SOLARIZE_OP_H_
