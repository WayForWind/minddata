
#include "OURSdata/dataset/engine/datasetops/pipeline_op.h"
#include <iostream>

namespace ours {
namespace dataset {
// Constructor
PipelineOp::PipelineOp(int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler)
    : DatasetOp(op_connector_size, sampler) {}

// A print method typically used for debugging
void PipelineOp::Print(std::ostream &out, bool show_all) const {
  // Summary 1-liner print
  if (!show_all) {
    // Call super class printer
    DatasetOp::Print(out, show_all);
    out << " [workers: ";
    if (this->inlined()) {
      out << "0 (inlined)]";
    } else {
      out << "1]";  // Pipeline ops only have 1 worker
    }
  } else {
    // Detailed print
    DatasetOp::Print(out, show_all);
    out << "\nNum workers: ";
    if (this->inlined()) {
      out << "0 (inlined)";
    } else {
      out << "1";  // Pipeline ops only have 1 worker
    }
  }
}
}  // namespace dataset
}  // namespace ours
