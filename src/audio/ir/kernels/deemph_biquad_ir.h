

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DEEMPH_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DEEMPH_BIQUAD_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kDeemphBiquadOperation[] = "DeemphBiquad";

class DeemphBiquadOperation : public TensorOperation {
 public:
  explicit DeemphBiquadOperation(int32_t sample_rate);

  ~DeemphBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDeemphBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DEEMPH_BIQUAD_IR_H_
