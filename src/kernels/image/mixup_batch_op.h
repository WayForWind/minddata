
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MIXUP_BATCH_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MIXUP_BATCH_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class MixUpBatchOp : public RandomTensorOp {
 public:
  explicit MixUpBatchOp(float alpha);

  ~MixUpBatchOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kMixUpBatchOp; }

 private:
  // a helper function to shorten the main Compute function
  Status ComputeLabels(const std::shared_ptr<Tensor> &label, std::shared_ptr<Tensor> *out_labels,
                       std::vector<int64_t> *rand_indx, const std::vector<int64_t> &label_shape, float lam,
                       size_t images_size);

  float alpha_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MIXUP_BATCH_OP_H_
