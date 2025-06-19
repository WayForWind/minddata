

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class RandomPosterizeOp : public RandomTensorOp {
 public:
  /// \brief Constructor
  /// \param[in] bit_range: Minimum and maximum bits in range
  explicit RandomPosterizeOp(const std::vector<uint8_t> &bit_range);

  ~RandomPosterizeOp() override = default;

  std::string Name() const override { return kRandomPosterizeOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// Member variables
 private:
  std::vector<uint8_t> bit_range_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_
