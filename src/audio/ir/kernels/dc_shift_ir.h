

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DC_SHIFT_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DC_SHIFT_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kDCShiftOperation[] = "DCShift";

class DCShiftOperation : public TensorOperation {
 public:
  DCShiftOperation(float shift, float limiter_gain);

  ~DCShiftOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDCShiftOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float shift_;
  float limiter_gain_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DC_SHIFT_IR_H_
