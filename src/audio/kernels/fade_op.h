

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FADE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FADE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class FadeOp : public TensorOp {
 public:
  /// Default fade in len to be used
  static const int32_t kFadeInLen;
  /// Default fade out len to be used
  static const int32_t kFadeOutLen;
  /// Default fade shape to be used
  static const FadeShape kFadeShape;

  explicit FadeOp(int32_t fade_in_len = kFadeInLen, int32_t fade_out_len = kFadeOutLen,
                  FadeShape fade_shape = kFadeShape)
      : fade_in_len_(fade_in_len), fade_out_len_(fade_out_len), fade_shape_(fade_shape) {}

  ~FadeOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFadeOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t fade_in_len_;
  int32_t fade_out_len_;
  FadeShape fade_shape_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FADE_OP_H_
