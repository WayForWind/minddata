
#include "OURSdata/dataset/kernels/image/normalize_pad_op.h"

#include <random>

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
NormalizePadOp::NormalizePadOp(std::vector<float> mean, std::vector<float> std, std::string dtype, bool is_hwc)
    : mean_(std::move(mean)), std_(std::move(std)), dtype_(std::move(dtype)), is_hwc_(is_hwc) {}

Status NormalizePadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the Normalization + pad
  return NormalizePad(input, output, mean_, std_, dtype_, is_hwc_);
}

void NormalizePadOp::Print(std::ostream &out) const {
  out << "NormalizePadOp, mean: ";
  for (const auto &m : mean_) {
    out << m << ", ";
  }
  out << "}" << std::endl << "std: ";
  for (const auto &s : std_) {
    out << s << ", ";
  }
  out << "}" << std::endl << "is_hwc: " << is_hwc_;
  out << "}" << std::endl;
}
}  // namespace dataset
}  // namespace ours
