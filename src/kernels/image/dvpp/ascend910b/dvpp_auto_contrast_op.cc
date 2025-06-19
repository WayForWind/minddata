/

#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_auto_contrast_op.h"

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

DvppAutoContrastOp::DvppAutoContrastOp(const std::vector<float> &cutoff, const std::vector<uint32_t> &ignore)
    : cutoff_(cutoff), ignore_(ignore) {}

Status DvppAutoContrastOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                   std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppAutoContrast: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppAutoContrastOp));
  constexpr int64_t kLengthMax = 256;
  CHECK_FAIL_RETURN_UNEXPECTED(ignore_.size() <= kLengthMax,
                               "DvppAutoContrast: the length of ignore should be less or equal to 256, but got: " +
                                 std::to_string(ignore_.size()));

  // run dvpp
  APP_ERROR ret = AclAdapter::GetInstance().DvppAutoContrast(input, output, cutoff_, ignore_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppAutoContrast: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
