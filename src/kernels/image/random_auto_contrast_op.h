
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomAutoContrastOp : public RandomTensorOp {
 public:
  RandomAutoContrastOp(float cutoff, const std::vector<uint32_t> &ignore, float prob)
      : cutoff_(cutoff), ignore_(ignore), distribution_(prob) {}

  ~RandomAutoContrastOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RandomAutoContrastOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomAutoContrastOp; }

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_
