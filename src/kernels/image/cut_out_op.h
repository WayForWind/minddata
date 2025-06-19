
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class CutOutOp : public RandomTensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const bool kDefRandomColor;

  // Constructor for CutOutOp
  // @param box_height box height
  // @param box_width box_width
  // @param num_patches how many patches to erase from image
  // @param random_color boolean value to indicate fill patch with random color
  // @param fill_colors value for the color to fill patch with
  // @param is_hwc Check if input is HWC/CHW format
  // @note maybe using unsigned long int isn't the best here according to our coding rules
  CutOutOp(int32_t box_height, int32_t box_width, int32_t num_patches, bool random_color = kDefRandomColor,
           std::vector<uint8_t> fill_colors = {}, bool is_hwc = true);

  ~CutOutOp() override = default;

  void Print(std::ostream &out) const override {
    out << "CutOut:: box_height: " << box_height_ << " box_width: " << box_width_ << " num_patches: " << num_patches_;
  }

  // Overrides the base class compute function
  // Calls the erase function in ImageUtils, this function takes an input tensor
  // and overwrites some of its data using openCV, the output memory is manipulated to contain the result
  // @return Status The status code returned
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kCutOutOp; }

 private:
  int32_t box_height_;
  int32_t box_width_;
  int32_t num_patches_;
  bool random_color_;
  std::vector<uint8_t> fill_colors_;
  bool is_hwc_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CUT_OUT_OP_H_
