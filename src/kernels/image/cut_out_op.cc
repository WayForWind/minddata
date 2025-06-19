
#include "OURSdata/dataset/kernels/image/cut_out_op.h"

#include <random>

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
const bool CutOutOp::kDefRandomColor = false;

// constructor
CutOutOp::CutOutOp(int32_t box_height, int32_t box_width, int32_t num_patches, bool random_color,
                   std::vector<uint8_t> fill_colors, bool is_hwc)
    : box_height_(box_height),
      box_width_(box_width),
      num_patches_(num_patches),
      random_color_(random_color),
      fill_colors_(std::move(fill_colors)),
      is_hwc_(is_hwc) {}

// main function call for cut out
Status CutOutOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::shared_ptr<CVTensor> inputCV = CVTensor::AsCVTensor(input);
  // cut out will clip the erasing area if the box is near the edge of the image and the boxes are black
  RETURN_IF_NOT_OK(CutOut(inputCV, output, box_height_, box_width_, num_patches_, false, random_color_,
                          &random_generator_, fill_colors_, is_hwc_));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
