
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_RGB_TO_BGR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_RGB_TO_BGR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RgbToBgrOp : public TensorOp {
 public:
  RgbToBgrOp() = default;

  ~RgbToBgrOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRgbToBgrOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_RGB_TO_BGR_OP_H_
