
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SLIDING_WINDOW_CMN_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SLIDING_WINDOW_CMN_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kSlidingWindowCmnOperation[] = "SlidingWindowCmn";

class SlidingWindowCmnOperation : public TensorOperation {
 public:
  SlidingWindowCmnOperation(int32_t cmn_window, int32_t min_cmn_window, bool center, bool norm_vars);

  ~SlidingWindowCmnOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSlidingWindowCmnOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t cmn_window_;
  int32_t min_cmn_window_;
  bool center_;
  bool norm_vars_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SLIDING_WINDOW_CMN_IR_H_
