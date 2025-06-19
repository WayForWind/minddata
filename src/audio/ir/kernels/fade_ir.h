

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kFadeOperation[] = "Fade";

class FadeOperation : public TensorOperation {
 public:
  FadeOperation(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape);

  ~FadeOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kFadeOperation; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t fade_in_len_;
  int32_t fade_out_len_;
  FadeShape fade_shape_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_
