

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_RESOURCE_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_RESOURCE_H_

#include <memory>
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/visible.h"
#include "OURSdata/dataset/core/device_tensor.h"
#include "OURSdata/dataset/core/tensor.h"

namespace ours {
namespace dataset {

class DeviceResource {
 public:
  DeviceResource() = default;

  virtual ~DeviceResource() = default;

  virtual Status InitResource(uint32_t device_id);

  virtual Status FinalizeResource();

  virtual Status Sink(const our::OURTensor &host_input, std::shared_ptr<DeviceTensor> *device_input);

  virtual Status Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output);

  virtual std::shared_ptr<void> GetInstance();

  virtual Status DeviceDataRelease();

  virtual void *GetContext();

  virtual void *GetStream();
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_DEVICE_RESOURCE_H
