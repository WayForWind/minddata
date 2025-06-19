
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TO_TENSOR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TO_TENSOR_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ToTensorOp : public TensorOp {
 public:
  explicit ToTensorOp(const DataType::Type &output_type) : output_type_(output_type) {}

  explicit ToTensorOp(const std::string &output_type) : output_type_(DataType(output_type)) {}

  ~ToTensorOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kToTensorOp; }

 private:
  DataType output_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TO_TENSOR_OP_H_
