

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_DETENSOR_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_DETENSOR_H_
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "include/api/status.h"
#include "include/api/visible.h"
#include "ir/api_tensor_impl.h"

namespace ours {
namespace dataset {
class Tensor;
class DeviceTensor;

class DETensor : public our::OURTensor::Impl {
 public:
  DETensor() = default;
  ~DETensor() = default;
  explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_impl);

  DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device);

  const std::string &Name() const override;

  enum our::DataType DataType() const override;

  size_t DataSize() const override;

  const std::vector<int64_t> &Shape() const override;
  void SetShape(const std::vector<int64_t> &shape) override { shape_ = shape; };

  int64_t ElementNum() const;

  std::shared_ptr<const void> Data() const override;

  void *MutableData() override;

  bool IsDevice() const override;

  std::shared_ptr<our::OURTensor::Impl> Clone() const override;

 private:
  std::shared_ptr<dataset::Tensor> tensor_impl_;
  std::shared_ptr<dataset::DeviceTensor> device_tensor_impl_;
  bool is_device_;
  std::string name_;
  enum our::DataType type_;
  std::vector<int64_t> shape_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_DETENSOR_H_
