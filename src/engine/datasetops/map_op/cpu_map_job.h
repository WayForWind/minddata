
#ifndef DATASET_ENGINE_DATASETOPS_MAP_OP_CPU_MAP_JOB_H_
#define DATASET_ENGINE_DATASETOPS_MAP_OP_CPU_MAP_JOB_H_

#include <memory>
#include <vector>
#include "OURSdata/dataset/engine/datasetops/map_op/map_job.h"

namespace ours {
namespace dataset {
class CpuMapJob : public MapJob {
 public:
  // Constructor
  CpuMapJob();

  // Constructor
  explicit CpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations);

  // Destructor
  ~CpuMapJob();

  // A pure virtual run function to execute a cpu map job
  Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) override;

#if defined(ENABLE_D)
  // A pure virtual run function to execute a npu map job for Ascend910B DVPP
  Status Run(std::vector<TensorRow> in, std::vector<TensorRow> *out, our::device::DeviceContext *device_context,
             const size_t &stream_id) override {
    RETURN_STATUS_UNEXPECTED("The run operation is not implemneted in CPU platform.");
  }
#endif

  MapTargetDevice Type() override { return MapTargetDevice::kCpu; }
};

}  // namespace dataset
}  // namespace ours

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_CPU_MAP_JOB_H_
