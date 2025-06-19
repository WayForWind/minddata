/

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_

#include <memory>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/util/status.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace ours {
namespace dataset {
class Tensor;
class DATASET_API DeviceTensorAscend910B {
 public:
  DeviceTensorAscend910B(const TensorShape &shape, const DataType &type, device::DeviceContext *device_context,
                         const size_t &stream_id, bool is_hwc = true);

  // create device_tensor by empty
  static Status CreateDeviceTensor(const TensorShape &shape, const DataType &type,
                                   device::DeviceContext *device_context, const size_t &stream_id,
                                   std::shared_ptr<DeviceTensorAscend910B> *out, bool is_hwc = true,
                                   std::vector<int> channels = {1, 3});

  // create device_tensor by host tensor
  static Status CreateDeviceTensor(std::shared_ptr<Tensor> tensor, device::DeviceContext *device_context,
                                   const size_t &stream_id, std::shared_ptr<DeviceTensorAscend910B> *out,
                                   bool is_hwc = true, std::vector<int> channels = {1, 3});

  ~DeviceTensorAscend910B();

  device::DeviceContext *GetDeviceContext() { return device_context_; }

  size_t GetStreamID() { return stream_id_; }

  void SetDeviceAddress(void *device_address) { device_address_ = device_address; }

  void *GetDeviceAddress() { return device_address_; }

  void SetDeviceTensor(void *tensor) { tensor_ = tensor; }

  TensorShape &GetShape() { return tensor_shape_; }

  DataType GetType() { return data_type_; }

  void *GetDeviceTensor() { return tensor_; }

  Status ToHostTensor(std::shared_ptr<Tensor> *host_tensor);

  bool AddWorkSpace(void *workspace);

  bool AddMaintenFloatArrayMemory(void *float_array);

  bool AddMaintenIntArrayMemory(void *int_array);

  bool ReleaseDeviceMemory();

 private:
  // Ascend910B resource
  device::DeviceContext *device_context_;
  size_t stream_id_;
  void *device_address_;
  void *tensor_;                      // aclTensor which point to device_address_
  void *workspace_;                   // used by step1 with dvpp HostAPI
  std::vector<void *> float_arrays_;  // used by dvpp in execution
  std::vector<void *> int_arrays_;    // used by dvpp in execution
  TensorShape tensor_shape_;
  DataType data_type_;
  bool is_hwc_;
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_ASCEND910B_H_
