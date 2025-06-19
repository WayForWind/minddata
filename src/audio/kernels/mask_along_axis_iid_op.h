/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_IID_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_IID_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class MaskAlongAxisIIDOp : public RandomTensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] mask_param Number of columns to be masked, will be uniformly sampled from [0, mask_param],
  ///     must be non negative.
  /// \param[in] mask_value Value to assign to the masked columns.
  /// \param[in] axis Axis to apply masking on (1 for frequency and 2 for time).
  MaskAlongAxisIIDOp(int32_t mask_param, float mask_value, int32_t axis)
      : mask_param_(mask_param), mask_value_(mask_value), axis_(axis) {}

  ~MaskAlongAxisIIDOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kMaskAlongAxisIIDOp; }

 private:
  int32_t mask_param_;
  float mask_value_;
  int32_t axis_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_IID_OP_H_
