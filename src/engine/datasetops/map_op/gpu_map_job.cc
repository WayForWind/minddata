

#include "OURSdata/dataset/engine/datasetops/map_op/gpu_map_job.h"

namespace ours {
namespace dataset {

// Constructor
GpuMapJob::GpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations) : MapJob(operations) {}

// Destructor
GpuMapJob::~GpuMapJob() = default;
}  // namespace dataset
}  // namespace ours
