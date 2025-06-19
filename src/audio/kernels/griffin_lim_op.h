/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GRIFFIN_LIM_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GRIFFIN_LIM_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
class GriffinLimOp : public RandomTensorOp {
 public:
  GriffinLimOp(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type,
               float power, float momentum, int32_t length, bool rand_init)
      : n_fft_(n_fft),
        n_iter_(n_iter),
        win_length_(win_length),
        hop_length_(hop_length),
        window_type_(window_type),
        power_(power),
        momentum_(momentum),
        length_(length),
        rand_init_(rand_init) {}

  ~GriffinLimOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kGriffinLimOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

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
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GRIFFIN_LIM_OP_H_
