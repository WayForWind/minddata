
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MATH_UTILS_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MATH_UTILS_H_

#include <memory>
#include <random>
#include <vector>

#include "OURSdata/dataset/util/status.h"

#define CV_PI 3.1415926535897932384626433832795

namespace ours {
namespace dataset {
/// \brief Returns lower and upper pth percentiles of the input histogram.
/// \param[in] hist: Input histogram (mutates the histogram for computation purposes)
/// \param[in] hi_p: Right side percentile
/// \param[in] low_p: Left side percentile
/// \param[out] hi: Value at high end percentile
/// \param[out] lo: Value at low end percentile
Status ComputeUpperAndLowerPercentiles(std::vector<int32_t> *hist, int32_t hi_p, int32_t low_p, int32_t *hi,
                                       int32_t *lo);

/// \brief Converts degrees input to radians.
/// \param[in] degrees: Input degrees
/// \param[out] radians_target: Radians output
Status DegreesToRadians(float_t degrees, float_t *radians_target);

/// \brief Generates a random real number in [a,b).
/// \param[in] a: Start of range
/// \param[in] b: End of range
/// \param[in] rnd: Random device
/// \param[out] result: Random number in range [a,b)
Status GenerateRealNumber(float_t a, float_t b, std::mt19937 *rnd, float_t *result);
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_MATH_UTILS_H_
