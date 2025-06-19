

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_AMPLITUDE_TO_DB_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_AMPLITUDE_TO_DB_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {

class AmplitudeToDBOp : public TensorOp {
 public:
  AmplitudeToDBOp(ScaleType stype, float ref_value, float amin, float top_db)
      : stype_(stype), ref_value_(ref_value), amin_(amin), top_db_(top_db) {}

  ~AmplitudeToDBOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAmplitudeToDBOp; }

 private:
  ScaleType stype_;
  float ref_value_;
  float amin_;
  float top_db_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_AMPLITUDE_TO_DB_OP_H_
