/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FILTFILT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FILTFILT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class FiltfiltOp : public TensorOp {
 public:
  FiltfiltOp(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

  ~FiltfiltOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": a_coeffs: ";
    for (size_t i = 0; i < a_coeffs_.size(); i++) {
      out << a_coeffs_[i] << " ";
    }
    out << "b_coeffs: ";
    for (size_t i = 0; i < b_coeffs_.size(); i++) {
      out << b_coeffs_[i] << " ";
    }
    out << "clamp: " << clamp_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFiltfiltOp; }

 private:
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_FILTFILT_OP_H_
