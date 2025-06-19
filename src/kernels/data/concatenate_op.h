

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_CONCATENATE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_CONCATENATE_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class ConcatenateOp : public TensorOp {
 public:
  /// Constructor to ConcatenateOp.
  /// @param int8_t axis - axis to concatenate tensors along.
  /// @param std::shared_ptr<Tensor> prepend - prepend tensor.
  /// @param std::shared_ptr<Tensor> append -append tensor.
  ConcatenateOp(int8_t axis, std::shared_ptr<Tensor> prepend, std::shared_ptr<Tensor> append)
      : axis_(axis), prepend_(std::move(prepend)), append_(std::move(append)) {}

  ~ConcatenateOp() override = default;

  /// Compute method allowing multiple tensors as inputs
  /// @param TensorRow &input - input tensor rows
  /// @param TensorRow *output - output tensor rows
  Status Compute(const TensorRow &input, TensorRow *output) override;

  /// Compute tensor output shape
  /// @param std::vector<TensorShape> &inputs - vector of input tensor shapes
  /// @param std::vector<TensorShape< &outputs - vector of output tensor shapes
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// Number of inputs the tensor operation accepts
  uint32_t NumInput() override { return 0; }

  std::string Name() const override { return kConcatenateOp; }

 private:
  int8_t axis_;
  std::shared_ptr<Tensor> prepend_;
  std::shared_ptr<Tensor> append_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CONCATENATE_OP_H
