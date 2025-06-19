

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_H_
#include <memory>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class Tensor;
class DATASET_API DeviceTensor : public Tensor {
 public:
  DeviceTensor(const TensorShape &shape, const DataType &type);

  ~DeviceTensor() override = default;

  Status SetAttributes(uint8_t *data_ptr, const uint32_t &dataSize, const uint32_t &width, const uint32_t &widthStride,
                       const uint32_t &height, const uint32_t &heightStride);

  static Status CreateEmpty(const TensorShape &shape, const DataType &type, std::shared_ptr<DeviceTensor> *out);

  static Status CreateFromDeviceMemory(const TensorShape &shape, const DataType &type, uint8_t *data_ptr,
                                       const uint32_t &dataSize, const std::vector<uint32_t> &attributes,
                                       std::shared_ptr<DeviceTensor> *out);

  const unsigned char *GetHostBuffer();

  const uint8_t *GetDeviceBuffer();

  uint8_t *GetDeviceMutableBuffer();

  std::vector<uint32_t> GetYuvStrideShape();

  uint32_t DeviceDataSize();

  DataType DeviceDataType() const;

  bool HasDeviceData() { return device_data_ != nullptr; }

 private:
  Status SetSize_(const uint32_t &new_size);

  Status SetYuvStrideShape_(const uint32_t &width, const uint32_t &widthStride, const uint32_t &height,
                            const uint32_t &heightStride);

  Status DataPop_(std::shared_ptr<Tensor> *host_tensor);

  std::vector<uint32_t> YUV_shape_;  // YUV_shape_ = {width, widthStride, height, heightStride}

  uint8_t *device_data_;

  uint32_t size_;

  DataType device_data_type_;

  // We use this Tensor to store device_data when DeviceTensor pop onto host
  std::shared_ptr<Tensor> host_data_tensor_;
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_DEVICE_TENSOR_H_
