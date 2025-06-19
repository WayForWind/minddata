
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class NormalizeOp : public TensorOp {
 public:
  NormalizeOp(std::vector<float> mean, std::vector<float> std, bool is_hwc);

  ~NormalizeOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kNormalizeOp; }

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  bool is_hwc_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_
