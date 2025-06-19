
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_EQUALIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_EQUALIZE_OP_H_

#include <memory>
#include <random>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomEqualizeOp : public RandomTensorOp {
 public:
  explicit RandomEqualizeOp(float prob) : distribution_(prob) {}

  ~RandomEqualizeOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RandomEqualizeOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomEqualizeOp; }

 private:
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_EQUALIZE_OP_H_
