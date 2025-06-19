ns under the License.
 */

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_AMPLITUDE_TO_DB_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_AMPLITUDE_TO_DB_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kAmplitudeToDBOperation[] = "AmplitudeToDB";

class AmplitudeToDBOperation : public TensorOperation {
 public:
  AmplitudeToDBOperation(ScaleType stype, float ref_value, float amin, float top_db);

  ~AmplitudeToDBOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  ScaleType stype_;
  float ref_value_;
  float amin_;
  float top_db_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_AMPLITUDE_TO_DB_IR_H_
