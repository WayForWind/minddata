/

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PERSPECTIVE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PERSPECTIVE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class PerspectiveOp : public TensorOp {
 public:
  /// Constructor
  PerspectiveOp(const std::vector<std::vector<int32_t>> &start_points,
                const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation);

  ~PerspectiveOp() override = default;

  std::string Name() const override { return kPerspectiveOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 protected:
  std::vector<std::vector<int32_t>> start_points_;
  std::vector<std::vector<int32_t>> end_points_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PERSPECTIVE_OP_H_
