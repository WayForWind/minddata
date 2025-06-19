/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class PadToSizeOp : public TensorOp {
 public:
  PadToSizeOp(std::vector<int32_t> size, std::vector<int32_t> offset, std::vector<uint8_t> fill_value,
              BorderType padding_mode);

  ~PadToSizeOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kPadToSizeOp; }

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> offset_;
  std::vector<uint8_t> fill_value_;
  BorderType boarder_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_
