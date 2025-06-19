
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_COMPUTE_DELTAS_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_COMPUTE_DELTAS_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kComputeDeltasOperation[] = "ComputeDeltas";

class ComputeDeltasOperation : public TensorOperation {
 public:
  ComputeDeltasOperation(int32_t win_length, BorderType pad_mode);

  ~ComputeDeltasOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kComputeDeltasOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t win_length_;
  BorderType pad_mode_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_COMPUTE_DELTAS_IR_H_
