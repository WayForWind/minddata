

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_POSTERIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_POSTERIZE_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class PosterizeOp : public TensorOp {
 public:
  /// \brief Constructor
  /// \param[in] bit: bits to use
  explicit PosterizeOp(uint8_t bit);

  ~PosterizeOp() override = default;

  std::string Name() const override { return kPosterizeOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  uint8_t bit_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_POSTERIZE_OP_H_
