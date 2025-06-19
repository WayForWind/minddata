

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DITHER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DITHER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DitherOp : public RandomTensorOp {
 public:
  DitherOp(DensityFunction density_function, bool noise_shaping)
      : density_function_(density_function), noise_shaping_(noise_shaping) {}

  ~DitherOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << " density_function: " << density_function_ << ", noise_shaping: " << noise_shaping_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kDitherOp; }

 private:
  DensityFunction density_function_;
  bool noise_shaping_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DITHER_OP_H_
