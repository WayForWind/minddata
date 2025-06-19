
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_

#include <random>
#include <utility>
#include <vector>

#include "lite_cv/lite_mat.h"
#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
/// \brief Calculate the coefficient for interpolation firstly
int calc_coeff(int input_size, int out_size, int input0, int input1, const struct interpolation *interp,
               std::vector<int> &regions, std::vector<double> &coeffs_interp);

/// \brief Normalize the coefficient for interpolation
void normalize_coeff(int out_size, int kernel_size, const std::vector<double> &prekk, std::vector<int> &kk);

/// \brief Apply horizontal interpolation on input image
Status ImagingHorizontalInterp(LiteMat &output, LiteMat input, int offset, int kernel_size,
                               const std::vector<int> &regions, const std::vector<double> &prekk);

/// \brief Apply Vertical interpolation on input image
Status ImagingVerticalInterp(LiteMat &output, LiteMat input, int offset, int kernel_size,
                             const std::vector<int> &regions, const std::vector<double> &prekk);

/// \brief Mainly logic of Cubic interpolation
bool ImageInterpolation(LiteMat input, LiteMat &output, int x_size, int y_size, struct interpolation *interp,
                        const int rect[4]);

/// \brief Apply cubic interpolation on input image and obtain the output image
/// \param[in] input Input image
/// \param[out] dst Output image
/// \param[in] dst_w expected Output image width
/// \param[in] dst_h expected Output image height
bool ResizeCubic(const LiteMat &input, const LiteMat &dst, int dst_w, int dst_h);
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_CUBIC_OP_H_
