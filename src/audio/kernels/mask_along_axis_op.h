/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class MaskAlongAxisOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] mask_start Starting position of the mask.
  /// \param[in] mask_width The width of the mask.
  /// \param[in] mask_value Value to assign to the masked columns.
  /// \param[in] axis Axis to apply masking on (1 for frequency and 2 for time).
  MaskAlongAxisOp(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis)
      : mask_start_(mask_start), mask_width_(mask_width), mask_value_(mask_value), axis_(axis) {}

  ~MaskAlongAxisOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kMaskAlongAxisOp; }

 private:
  int32_t mask_start_;
  int32_t mask_width_;
  float mask_value_;
  int32_t axis_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MASK_ALONG_AXIS_OP_H_
