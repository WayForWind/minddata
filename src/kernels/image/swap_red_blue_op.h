
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class SwapRedBlueOp : public TensorOp {
 public:
  /// \brief Constructor
  SwapRedBlueOp() = default;

  SwapRedBlueOp(const SwapRedBlueOp &rhs) = default;

  SwapRedBlueOp(SwapRedBlueOp &&rhs) = default;

  ~SwapRedBlueOp() override = default;

  void Print(std::ostream &out) const override { out << "SwapRedBlueOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSwapRedBlueOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_SWAP_RED_BLUE_OP_H_
