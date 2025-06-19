
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ERASE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ERASE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class EraseOp : public TensorOp {
 public:
  EraseOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value,
          bool inplace = false);

  ~EraseOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kEraseOp; }

 private:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  std::vector<float> value_;
  bool inplace_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_ERASE_OP_H_
