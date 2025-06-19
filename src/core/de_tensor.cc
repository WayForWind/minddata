

#include "OURSdata/dataset/core/de_tensor.h"
#include "OURSdata/dataset/core/device_tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/core/type_id.h"
#include "OURSdata/dataset/util/log_adapter.h"
#define EXCEPTION_IF_NULL(ptr) MS_EXCEPTION_IF_NULL(ptr)

namespace ours {
namespace dataset {

DETensor::DETensor(std::shared_ptr<dataset::Tensor> tensor_impl)
    : tensor_impl_(tensor_impl),
      name_("MindDataTensor"),
      type_(static_cast<our::DataType>(DETypeToMSType(tensor_impl_->type()))),
      shape_(tensor_impl_->shape().AsVector()),
      is_device_(false) {}

DETensor::DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device)
    : device_tensor_impl_(device_tensor_impl), name_("MindDataDeviceTensor"), is_device_(is_device) {
  // The sequence of shape_ is (width, widthStride, height, heightStride) in Dvpp module
  // We need to add [1]widthStride and [3]heightStride, which are actual YUV image shape, into shape_ attribute
  if (device_tensor_impl && device_tensor_impl->GetYuvStrideShape().size() > 0) {
    uint8_t flag = 0;
    for (auto &i : device_tensor_impl->GetYuvStrideShape()) {
      if (flag % 2 == 1) {
        int64_t j = static_cast<int64_t>(i);
        shape_.emplace_back(j);
      }
      ++flag;
    }
    std::reverse(shape_.begin(), shape_.end());
  }
  MS_LOG(INFO) << "This is a YUV420 format image, one pixel takes 1.5 bytes. Therefore, the shape of"
               << " image is in (H, W) format. You can search for more information about YUV420 format";
}

const std::string &DETensor::Name() const { return name_; }

enum our::DataType DETensor::DataType() const {
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return static_cast<our::DataType>(DETypeToMSType(device_tensor_impl_->DeviceDataType()));
  }
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<our::DataType>(DETypeToMSType(tensor_impl_->type()));
}

size_t DETensor::DataSize() const {
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return device_tensor_impl_->DeviceDataSize();
  }
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<size_t>(tensor_impl_->SizeInBytes());
}

const std::vector<int64_t> &DETensor::Shape() const { return shape_; }

int64_t DETensor::ElementNum() const {
  if (shape_.empty()) {
    // element number of scalar is 1
    return 1;
  }
  return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
}

std::shared_ptr<const void> DETensor::Data() const {
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return std::shared_ptr<const void>(device_tensor_impl_->GetHostBuffer(), [](const void *) {});
  }
  return std::shared_ptr<const void>(tensor_impl_->GetBuffer(), [](const void *) {});
}

void *DETensor::MutableData() {
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return static_cast<void *>(device_tensor_impl_->GetDeviceMutableBuffer());
  }
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<void *>(tensor_impl_->GetMutableBuffer());
}

bool DETensor::IsDevice() const { return is_device_; }

std::shared_ptr<our::OURTensor::Impl> DETensor::Clone() const {
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return std::make_shared<DETensor>(device_tensor_impl_, is_device_);
  }
  return std::make_shared<DETensor>(tensor_impl_);
}
}  // namespace dataset
}  // namespace ours
