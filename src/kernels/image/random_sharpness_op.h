
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomSharpnessOp : public RandomTensorOp {
 public:
  /// Adjust the sharpness of the input image by a random degree within the given range.
  /// \@param[in] start_degree A float indicating the beginning of the range.
  /// \@param[in] end_degree A float indicating the end of the range.
  explicit RandomSharpnessOp(float start_degree, float end_degree);

  ~RandomSharpnessOp() override = default;

  void Print(std::ostream &out) const override { out << Name(); }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomSharpnessOp; }

 protected:
  float start_degree_;
  float end_degree_;
  std::uniform_real_distribution<float> distribution_{-1.0, 1.0};
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_SHARPNESS_OP_H_
