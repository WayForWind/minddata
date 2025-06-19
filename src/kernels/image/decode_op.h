
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DecodeOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const bool kDefRgbFormat;

  explicit DecodeOp(bool rgb = true);

  ~DecodeOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDecodeOp; }

 private:
  bool is_rgb_format_ = true;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_OP_H_
