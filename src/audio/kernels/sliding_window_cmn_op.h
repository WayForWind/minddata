
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SLIDING_WINDOW_CMN_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SLIDING_WINDOW_CMN_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/audio/kernels/audio_utils.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class SlidingWindowCmnOp : public TensorOp {
 public:
  /// \brief Constructor of SlidingWindowCmnOp.
  /// \param[in] cmn_window - The window in frames for running average CMN computation.
  /// \param[in] min_cmn_window - The minimum CMN window. Only applicable if center is false, ignored if center==true.
  /// \param[in] center - If true, use a window centered on the current frame. If false, window is to the left.
  /// \param[in] norm_vars - If true, normalize variance to one.
  SlidingWindowCmnOp(int32_t cmn_window, int32_t min_cmn_window, bool center, bool norm_vars)
      : cmn_window_(cmn_window), min_cmn_window_(min_cmn_window), center_(center), norm_vars_(norm_vars) {}

  /// \brief Destructor of SlidingWindowCmnOp.
  ~SlidingWindowCmnOp() override = default;

  /// \brief Perform sliding window CMN to tensor.
  /// \param[in] input - Input tensor of Op.
  /// \param[out] output - Output tensor of Op.
  /// \return Status code.
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// \brief Print name of op.
  std::string Name() const override { return kSlidingWindowCmnOp; }

 private:
  int32_t cmn_window_;      // The window in frames for running average CMN computation.
  int32_t min_cmn_window_;  // The minimum CMN window. Only applicable if center == false, ignored if center==true.
  bool center_;             // If true, use a window centered on the current frame. If false, window is to the left.
  bool norm_vars_;          // If true, normalize variance to one.
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SLIDING_WINDOW_CMN_OP_H_
