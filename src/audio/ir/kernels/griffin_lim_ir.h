/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kGriffinLimOperation[] = "GriffinLim";

class GriffinLimOperation : public TensorOperation {
 public:
  GriffinLimOperation(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type,
                      float power, float momentum, int32_t length, bool rand_init);

  ~GriffinLimOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_fft_;
  int32_t n_iter_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_type_;
  float power_;
  float momentum_;
  int32_t length_;
  bool rand_init_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_GRIFFIN_LIM_IR_H_
