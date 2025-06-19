/
#ifndef DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_
#define DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_

#include <memory>
#include <vector>
#include "OURSdata/dataset/engine/datasetops/map_op/map_job.h"
#include "runtime/hardware/device_context.h"

namespace ours {
namespace dataset {
class NpuMapJob : public MapJob {
 public:
  // Constructor
  NpuMapJob();

  // Constructor
  explicit NpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations);

  // Destructor
  ~NpuMapJob();

  // A pure virtual run function to execute a npu map job
  Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) override {
    RETURN_STATUS_UNEXPECTED("The run operation is not implemneted in NPU platform.");
  }

  // A pure virtual run function to execute a npu map job for Ascend910B DVPP
  Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out, our::device::DeviceContext *device_context,
             const size_t &stream_id) override;

  MapTargetDevice Type() override { return MapTargetDevice::kAscend910B; }
};

}  // namespace dataset
}  // namespace ours

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_NPU_MAP_JOB_H_
