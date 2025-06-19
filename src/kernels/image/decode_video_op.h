/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_VIDEO_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_VIDEO_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DecodeVideoOp : public TensorOp {
 public:
  DecodeVideoOp();

  ~DecodeVideoOp() = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDecodeVideoOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DECODE_VIDEO_OP_H_
