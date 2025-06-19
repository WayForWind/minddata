
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NORMALIZE_UTF8_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NORMALIZE_UTF8_OP_H_
#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class NormalizeUTF8Op : public TensorOp {
 public:
  static const NormalizeForm kDefNormalizeForm;
  explicit NormalizeUTF8Op(NormalizeForm normalize_form = kDefNormalizeForm) : normalize_form_(normalize_form) {}

  ~NormalizeUTF8Op() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kNormalizeUTF8Op; }

 private:
  NormalizeForm normalize_form_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NORMALIZE_UTF8_OP_H_
