
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class SlicePatchesOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefNumH;
  static const int32_t kDefNumW;
  static const uint8_t kDefFillV;
  static const SliceMode kDefSliceMode;

  explicit SlicePatchesOp(int32_t num_height = kDefNumH, int32_t num_width = kDefNumW,
                          SliceMode slice_mode = kDefSliceMode, uint8_t fill_value = kDefFillV);

  ~SlicePatchesOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << " patches number on height: " << num_height_ << ", patches number on width: " << num_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kSlicePatchesOp; }

 protected:
  int32_t num_height_;    // number of patches on height axis
  int32_t num_width_;     // number of patches on width axis
  SliceMode slice_mode_;  // PadModel, DropModel
  uint8_t fill_value_;    // border width in number of pixels in right and bottom direction
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_SLICE_PATCHES_OP_H_
