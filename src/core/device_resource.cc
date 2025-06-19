

#include "OURSdata/dataset/core/device_resource.h"

namespace ours {
namespace dataset {

Status DeviceResource::InitResource(uint32_t) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device? If yes, please implement this InitResource() in the derived class.");
}

Status DeviceResource::FinalizeResource() {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device? If yes, please implement this FinalizeResource() in the derived class.");
}

Status DeviceResource::Sink(const our::OURTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device whose device memory is available? If yes, please implement this Sink() in the "
                "derived class.");
}

Status DeviceResource::Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device whose device memory is available? If yes, please implement this Pop() in the "
                "derived class.");
}

Status DeviceResource::DeviceDataRelease() {
  return Status(
    StatusCode::kMDUnexpectedError,
    "Is this a valid device whose device memory is available? If yes, please implement this DeviceDataRelease() in the "
    "derived class.");
}

std::shared_ptr<void> DeviceResource::GetInstance() {
  MS_LOG(ERROR) << "Is this a device which contains a processor object? If yes, please implement this GetInstance() in "
                   "the derived class";
  return nullptr;
}

void *DeviceResource::GetContext() {
  MS_LOG(ERROR)
    << "Is this a device which contains context resource? If yes, please implement GetContext() in the derived class";
  return nullptr;
}

void *DeviceResource::GetStream() {
  MS_LOG(ERROR)
    << "Is this a device which contains stream resource? If yes, please implement GetContext() in the derived class";
  return nullptr;
}

}  // namespace dataset
}  // namespace ours
