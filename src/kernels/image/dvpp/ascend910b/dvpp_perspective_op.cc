/

#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_perspective_op.h"

#include "OURSdata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
constexpr int64_t h_lb = 6;     // height lower bound
constexpr int64_t h_ub = 8192;  // height upper bound
constexpr int64_t w_lb = 10;    // width lower bound
constexpr int64_t w_ub = 4096;  // width upper bound

Status DvppPerspectiveOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                  std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // check the input tensor shape
  if (input->GetShape().Rank() != kNHWCImageRank) {
    RETURN_STATUS_UNEXPECTED("DvppPerspective: invalid input shape, only support NHWC input, got rank: " +
                             std::to_string(input->GetShape().Rank()));
  }

  // the channel should be 3 or 1
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[kChannelIndexNHWC] == kMinImageChannel ||
                                 input->GetShape().AsVector()[kChannelIndexNHWC] == kDefaultImageChannel,
                               "DvppPerspective: the channel of the input is not 1 or 3.");

  // check Perspective support InterpolationMode
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour) {
    auto error = "DvppPerspective: Invalid interpolation mode, only support BILINEAR and NEAREST.";
    RETURN_STATUS_UNEXPECTED(error);
  }

  // the type should be uint8 or float
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetType() == DataType::DE_UINT8 || input->GetType() == DataType::DE_FLOAT32,
                               "DvppPerspective: the type of the input is not uint8 or float.");

  // Dvpp Limit
  int64_t input_h = input->GetShape()[kHeightIndexNHWC];
  int64_t input_w = input->GetShape()[kWidthIndexNHWC];
  RETURN_IF_NOT_OK(CheckDvppLimit(input_h, input_w, h_lb, w_lb, h_ub, w_ub, kDvppPerspectiveOp));

  APP_ERROR ret = AclAdapter::GetInstance().DvppPerspective(input, output, start_points_, end_points_, interpolation_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppPerspective: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
