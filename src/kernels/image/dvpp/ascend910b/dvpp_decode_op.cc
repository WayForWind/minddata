

#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_decode_op.h"

#include "OURSdata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status DvppDecodeOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // check the input tensor shape
  if (input->GetShape().Rank() != 1) {
    RETURN_STATUS_UNEXPECTED("DvppDecode: invalid input shape, only support 1D input, got rank: " +
                             std::to_string(input->GetShape().Rank()));
  }

  APP_ERROR ret = AclAdapter::GetInstance().DvppDecode(input, output);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppDecode: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}

Status DvppDecodeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, -1, 3});  // we don't know what is output image size, but we know it should be 3 channels
  if (inputs[0].Rank() == 1) {
    (void)outputs.emplace_back(out);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(),
                               "DvppDecode: invalid input shape, expected 1D input, but got input dimension is:" +
                                 std::to_string(inputs[0].Rank()));
  return Status::OK();
}

Status DvppDecodeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "DvppDecode: inputs cannot be empty.");
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = DataType(DataType::DE_UINT8);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
