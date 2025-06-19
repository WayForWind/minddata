
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ADJUST_SHARPNESS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ADJUST_SHARPNESS_OP_H_

#include <memory>
#include <random>
#include <string>

#include "OURSdata/dataset/kernels/image/sharpness_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomAdjustSharpnessOp : public RandomTensorOp {
 public:
  RandomAdjustSharpnessOp(float degree, float prob) : degree_(degree), distribution_(prob) {}

  ~RandomAdjustSharpnessOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RandomAdjustSharpnessOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomAdjustSharpnessOp; }

 private:
  float degree_;
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ADJUST_SHARPNESS_OP_H_
