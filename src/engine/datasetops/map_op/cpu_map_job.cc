

#include <set>
#include <string>
#include <utility>
#include "OURSdata/dataset/engine/datasetops/map_op/cpu_map_job.h"

namespace ours {
namespace dataset {

// Constructor
CpuMapJob::CpuMapJob() = default;

// Constructor
CpuMapJob::CpuMapJob(std::vector<std::shared_ptr<TensorOp>> operations) : MapJob(std::move(operations)) {}

// Destructor
CpuMapJob::~CpuMapJob() = default;

// A function to execute a cpu map job
Status CpuMapJob::Run(std::vector<TensorRow> in, std::vector<TensorRow> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  int32_t num_rows = in.size();
  for (int32_t row = 0; row < num_rows; row++) {
    TensorRow input_row = in[row];
    TensorRow result_row;
    for (size_t i = 0; i < ops_.size(); i++) {
      // Call compute function for cpu
      Status rc = ops_[i]->Compute(input_row, &result_row);
      if (rc.IsError()) {
        std::string op_name = ops_[i]->Name();
        RETURN_IF_NOT_OK(util::RebuildMapErrorMsg(input_row, op_name, &rc));
      }

      // Assign result_row to to_process for the next TensorOp processing, except for the last TensorOp in the list.
      if (i + 1 < ops_.size()) {
        input_row = std::move(result_row);
      }
    }
    out->push_back(std::move(result_row));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
