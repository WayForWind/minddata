
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_ENCODING_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_ENCODING_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kMuLawEncodingOperation[] = "MuLawEncoding";

class MuLawEncodingOperation : public TensorOperation {
 public:
  explicit MuLawEncodingOperation(int32_t quantization_channels);

  ~MuLawEncodingOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t quantization_channels_;
};  // class MuLawEncodingOperation
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MU_LAW_ENCODING_IR_H_
