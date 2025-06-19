

#include "OURSdata/dataset/kernels/image/random_color_adjust_op.h"

#include <random>

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
RandomColorAdjustOp::RandomColorAdjustOp(float s_bright_factor, float e_bright_factor, float s_contrast_factor,
                                         float e_contrast_factor, float s_saturation_factor, float e_saturation_factor,
                                         float s_hue_factor, float e_hue_factor)
    : bright_factor_start_(s_bright_factor),
      bright_factor_end_(e_bright_factor),
      contrast_factor_start_(s_contrast_factor),
      contrast_factor_end_(e_contrast_factor),
      saturation_factor_start_(s_saturation_factor),
      saturation_factor_end_(e_saturation_factor),
      hue_factor_start_(s_hue_factor),
      hue_factor_end_(e_hue_factor) {}

Status RandomColorAdjustOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "RandomColorAdjust", {}, {3}, {3}));

  // randomly select an augmentation to apply to the input image until all the transformations run
  std::vector<std::string> params_vector = {"brightness", "contrast", "saturation", "hue"};

  std::shuffle(params_vector.begin(), params_vector.end(), random_generator_);

  *output = std::static_pointer_cast<Tensor>(input);
  // determine if certain augmentation needs to be executed:
  for (const auto &param : params_vector) {
    // case switch
    if (param == "brightness") {
      if (CmpFloat(bright_factor_start_, bright_factor_end_) && CmpFloat(bright_factor_start_, 1.0f)) {
        MS_LOG(DEBUG) << "Not running brightness.";
      } else {
        // adjust the brightness of an image
        float random_factor =
          std::uniform_real_distribution<float>(bright_factor_start_, bright_factor_end_)(random_generator_);
        RETURN_IF_NOT_OK(AdjustBrightness(*output, output, random_factor));
      }
    } else if (param == "contrast") {
      if (CmpFloat(contrast_factor_start_, contrast_factor_end_) && CmpFloat(contrast_factor_start_, 1.0f)) {
        MS_LOG(DEBUG) << "Not running contrast.";
      } else {
        float random_factor =
          std::uniform_real_distribution<float>(contrast_factor_start_, contrast_factor_end_)(random_generator_);
        RETURN_IF_NOT_OK(AdjustContrast(*output, output, random_factor));
      }
    } else if (param == "saturation") {
      // adjust the Saturation of an image
      if (CmpFloat(saturation_factor_start_, saturation_factor_end_) && CmpFloat(saturation_factor_start_, 1.0f)) {
        MS_LOG(DEBUG) << "Not running saturation.";
      } else {
        float random_factor =
          std::uniform_real_distribution<float>(saturation_factor_start_, saturation_factor_end_)(random_generator_);
        RETURN_IF_NOT_OK(AdjustSaturation(*output, output, random_factor));
      }
    } else if (param == "hue") {
      if (CmpFloat(hue_factor_start_, hue_factor_end_) && CmpFloat(hue_factor_start_, 0.0f)) {
        MS_LOG(DEBUG) << "Not running hue.";
      } else {
        // adjust the Hue of an image
        float random_factor =
          std::uniform_real_distribution<float>(hue_factor_start_, hue_factor_end_)(random_generator_);
        RETURN_IF_NOT_OK(AdjustHue(*output, output, random_factor));
      }
    }
  }
  // now after we do all the transformations, the last one is fine
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
