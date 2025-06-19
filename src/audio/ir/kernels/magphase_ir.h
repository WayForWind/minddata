

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MAGPHASE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MAGPHASE_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kMagphaseOperation[] = "Magphase";

class MagphaseOperation : public TensorOperation {
 public:
  explicit MagphaseOperation(float power);

  ~MagphaseOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  std::string Name() const override { return kMagphaseOperation; }

  Status ValidateParams() override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  float power_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MAGPHASE_IR_H_
