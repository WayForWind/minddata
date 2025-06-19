
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FREQUENCY_MASKING_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FREQUENCY_MASKING_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class FrequencyMaskingOp : public RandomTensorOp {
 public:
  FrequencyMaskingOp(bool iid_masks = false, int32_t frequency_mask_param = 0, int32_t mask_start = 0,
                     float mask_value_ = 0.0);

  ~FrequencyMaskingOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kFrequencyMaskingOp; }

 private:
  bool iid_masks_;
  int32_t frequency_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FREQUENCY_MASKING_OP_H_
