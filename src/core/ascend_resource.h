

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_ASCEND_RESOURCE_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_ASCEND_RESOURCE_H_

#include <memory>
#include <string>
#include "OURSdata/dataset/core/device_resource.h"
#include "OURSdata/dataset/core/device_tensor.h"
#include "OURSdata/dataset/core/tensor.h"

namespace ours {
namespace dataset {

class AscendResource : public DeviceResource {
 public:
  AscendResource() = default;
  ~AscendResource() = default;

  Status InitResource(uint32_t device_id) override;

  Status FinalizeResource() override;

  Status Sink(const our::OURTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) override;

  Status Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) override;

  std::shared_ptr<void> GetInstance() override;

  Status DeviceDataRelease() override;

  void *GetContext() override;

  void *GetStream() override;

 private:
  std::shared_ptr<void> processor_;
};

}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_ASCEND_RESOURCE_H_
