
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_DECODING_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_DECODING_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kMuLawDecodingOperation[] = "MuLawDecoding";

class MuLawDecodingOperation : public TensorOperation {
 public:
  explicit MuLawDecodingOperation(int32_t quantization_channels);

  ~MuLawDecodingOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t quantization_channels_;
};  // class MuLawDecodingOperation
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_DECODING_IR_H_
