

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_

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
class RandomLightingOp : public RandomTensorOp {
 public:
  explicit RandomLightingOp(float alpha) : dist_(0, alpha) {}

  ~RandomLightingOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> *out) override;

  std::string Name() const override { return kRandomLightingOp; }

 private:
  std::normal_distribution<float> dist_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_LIGHTING_OP_H_
