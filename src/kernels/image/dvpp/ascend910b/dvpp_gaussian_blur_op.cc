/
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_gaussian_blur_op.h"

#include <vector>

#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
constexpr int64_t h_lb = 4;     // height lower bound
constexpr int64_t h_ub = 8192;  // height upper bound
constexpr int64_t w_lb = 6;     // width lower bound
constexpr int64_t w_ub = 4096;  // width upper bound

Status DvppGaussianBlurOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                   std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppGaussianBlur: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  std::vector<int64_t> kernel_size = {static_cast<int64_t>(kernel_x_), static_cast<int64_t>(kernel_y_)};
  std::vector<float> sigma = {sigma_x_, sigma_y_};
  uint32_t padding_mode = static_cast<float>(BorderType::kReflect);

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppGaussianBlurOp));

  const auto kKernelSizeOne = 1;
  const auto kKernelSizeThree = 3;
  const auto kKernelSizeFive = 5;
  for (const int64_t &k : kernel_size) {
    if (k != kKernelSizeOne && k != kKernelSizeThree && k != kKernelSizeFive) {
      auto error =
        "DvppGaussianBlur: the value of gaussian kernel only supports [1, 3, 5], but got " + std::to_string(k);
      RETURN_STATUS_UNEXPECTED(error);
    }
  }

  // run dvpp
  APP_ERROR ret = AclAdapter::GetInstance().DvppGaussianBlur(input, output, kernel_size, sigma, padding_mode);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppGaussianBlur: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
