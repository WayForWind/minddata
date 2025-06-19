
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_RGBA_TO_RGB_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_RGBA_TO_RGB_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RgbaToRgbOp : public TensorOp {
 public:
  RgbaToRgbOp() = default;

  ~RgbaToRgbOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRgbaToRgbOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_RGBA_TO_RGB_OP_H_
