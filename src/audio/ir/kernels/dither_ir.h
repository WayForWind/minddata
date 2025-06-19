

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DITHER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DITHER_IR_H_

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
constexpr char kDitherOperation[] = "Dither";

class DitherOperation : public TensorOperation {
 public:
  DitherOperation(DensityFunction density_function, bool noise_shaping);

  ~DitherOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDitherOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  DensityFunction density_function_;
  bool noise_shaping_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DITHER_IR_H_
