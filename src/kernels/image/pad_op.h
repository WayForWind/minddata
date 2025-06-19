
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class PadOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const BorderType kDefBorderType;
  static const uint8_t kDefFillR;
  static const uint8_t kDefFillG;
  static const uint8_t kDefFillB;

  // Constructor for PadOp.
  // @param pad_top number of pixels to pad the top of image with.
  // @param pad_bottom number of pixels to pad the bottom of the image with.
  // @param pad_left number of pixels to pad the left of the image with.
  // @param pad_right number of pixels to pad the right of the image with.
  // @param padding_mode BorderType enum, the type of boarders that we are using.
  // @param fill_r R value for the color to pad with.
  // @param fill_g G value for the color to pad with.
  // @param fill_b B value for the color to pad with.
  PadOp(int32_t pad_top, int32_t pad_bottom, int32_t pad_left, int32_t pad_right, BorderType padding_mode,
        uint8_t fill_r = kDefFillR, uint8_t fill_g = kDefFillG, uint8_t fill_b = kDefFillB);

  ~PadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kPadOp; }

 private:
  int32_t pad_top_;
  int32_t pad_bottom_;
  int32_t pad_left_;
  int32_t pad_right_;
  BorderType boarder_type_;
  uint8_t fill_r_;
  uint8_t fill_g_;
  uint8_t fill_b_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_OP_H_
