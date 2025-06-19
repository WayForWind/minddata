/

#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_adjust_sharpness_op.h"

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

Status DvppAdjustSharpnessOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                      std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // check the input tensor shape
  if (input->GetShape().Rank() != kNHWCImageRank) {
    RETURN_STATUS_UNEXPECTED("DvppAdjustSharpness: the input tensor is not HW, HWC or 1HWC, but got: " +
                             std::to_string(input->GetShape().Rank()));
  }

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppAdjustSharpnessOp));

  APP_ERROR ret = AclAdapter::GetInstance().DvppAdjustSharpness(input, output, factor_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppAdjustSharpness: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}

Status DvppAdjustSharpnessOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  return Status::OK();
}

Status DvppAdjustSharpnessOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
