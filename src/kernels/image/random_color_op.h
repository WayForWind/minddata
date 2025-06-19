

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_COLOR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_COLOR_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
/// \class RandomColorOp random_color_op.h
/// \brief Blends an image with its grayscale version with random weights
///        t and 1 - t generated from a given range.
///        If the range is trivial then the weights are determinate and
///        t equals the bound of the interval
class RandomColorOp : public RandomTensorOp {
 public:
  RandomColorOp() = default;

  ~RandomColorOp() override = default;

  /// \brief Constructor
  /// \param[in] t_lb lower bound for the random weights
  /// \param[in] t_ub upper bound for the random weights
  RandomColorOp(float t_lb, float t_ub);

  /// \brief the main function performing computations
  /// \param[in] in 2- or 3- dimensional tensor representing an image
  /// \param[out] out 2- or 3- dimensional tensor representing an image
  /// with the same dimensions as in
  Status Compute(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> *out) override;

  /// \brief returns the name of the op
  std::string Name() const override { return kRandomColorOp; }

 private:
  std::uniform_real_distribution<float> dist_;
  float t_lb_;
  float t_ub_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_RANDOM_COLOR_OP_H_
