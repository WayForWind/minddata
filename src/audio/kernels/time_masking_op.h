
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_MASKING_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_MASKING_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class TimeMaskingOp : public RandomTensorOp {
 public:
  TimeMaskingOp(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value);

  ~TimeMaskingOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kTimeMaskingOp; }

 private:
  bool iid_masks_;
  int32_t time_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_MASKING_OP_H_
