
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_PAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_PAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class NormalizePadOp : public TensorOp {
 public:
  NormalizePadOp(std::vector<float> mean, std::vector<float> std, std::string dtype = "float32", bool is_hwc = true);

  ~NormalizePadOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kNormalizePadOp; }

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  std::string dtype_;
  bool is_hwc_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_NORMALIZE_PAD_OP_H_
