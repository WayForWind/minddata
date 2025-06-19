

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_FILL_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_FILL_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class FillOp : public TensorOp {
 public:
  explicit FillOp(std::shared_ptr<Tensor> fill_value) : fill_value_(std::move(fill_value)) {}

  ~FillOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFillOp; }

 private:
  std::shared_ptr<Tensor> fill_value_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_FILL_OP_H_
