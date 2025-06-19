

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RIAA_BIQUAD_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RIAA_BIQUAD_IR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kRiaaBiquadOperation[] = "RiaaBiquad";

class RiaaBiquadOperation : public TensorOperation {
 public:
  explicit RiaaBiquadOperation(int32_t sample_rate);

  ~RiaaBiquadOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRiaaBiquadOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
};  // class RiaaBiquadOperation
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RIAA_BIQUAD_IR_H_
