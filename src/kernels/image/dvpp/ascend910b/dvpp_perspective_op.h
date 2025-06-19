/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PERSPECTIVE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PERSPECTIVE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DvppPerspectiveOp : public TensorOp {
 public:
  DvppPerspectiveOp(const std::vector<std::vector<int32_t>> &start_points,
                    const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation)
      : start_points_(start_points), end_points_(end_points), interpolation_(interpolation) {}

  ~DvppPerspectiveOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppPerspectiveOp; }

  bool IsDvppOp() override { return true; }

 protected:
  std::vector<std::vector<int32_t>> start_points_;
  std::vector<std::vector<int32_t>> end_points_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_PERSPECTIVE_OP_H_
