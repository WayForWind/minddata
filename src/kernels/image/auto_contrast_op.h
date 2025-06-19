

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_CONTRAST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_CONTRAST_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class AutoContrastOp : public TensorOp {
 public:
  /// Default cutoff to be used
  static const float kCutOff;
  /// Default ignore to be used
  static const std::vector<uint32_t> kIgnore;

  AutoContrastOp(const float &cutoff, const std::vector<uint32_t> &ignore) : cutoff_(cutoff), ignore_(ignore) {}

  ~AutoContrastOp() override = default;

  /// Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const AutoContrastOp &so) {
    so.Print(out);
    return out;
  }

  std::string Name() const override { return kAutoContrastOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  float cutoff_;
  std::vector<uint32_t> ignore_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_CONTRAST_OP_H_
