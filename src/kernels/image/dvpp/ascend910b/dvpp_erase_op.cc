/
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_erase_op.h"

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

Status DvppEraseOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                            std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);

  // the input should be NHWC, N is 1.
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->GetShape().Rank() == kNHWCImageRank,
    "DvppErase: the input tensor is not HW, HWC or 1HWC, but got: " + std::to_string(input->GetShape().Rank()));

  // Dvpp Limit
  std::vector<dsize_t> size = {input->GetShape().AsVector()[kHeightIndexNHWC],
                               input->GetShape().AsVector()[kWidthIndexNHWC]};
  int32_t input_h = size[kHeightIndex];
  int32_t input_w = size[kWidthIndex];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppEraseOp, "input"));

  if (input->GetType() == DataType::DE_FLOAT32) {
    for (const float &val : value_) {
      if (val > 1.) {
        std::string error = "When The input data is float32, the range of value should be [0, 1]";
        RETURN_STATUS_UNEXPECTED(error);
      }
    }
  }

  // Ensure that the dvpp erase operator does not report an error when its argument `value` is the default value
  if (value_.size() == 1) {
    std::vector<float> val(input->GetShape().AsVector()[kChannelIndexNHWC], value_[0]);
    value_ = val;
  }
  if (input->GetShape().AsVector()[kChannelIndexNHWC] != value_.size()) {
    std::string error = "The length of value should be the same as the value of channel";
    RETURN_STATUS_UNEXPECTED(error);
  }

  APP_ERROR ret = AclAdapter::GetInstance().DvppErase(input, output, top_, left_, height_, width_, value_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppErase: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
