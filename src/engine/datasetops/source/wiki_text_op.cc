

#include "OURSdata/dataset/engine/datasetops/source/wiki_text_op.h"

#include "include/common/debug/common.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/engine/datasetops/source/io_block.h"
#include "OURSdata/dataset/engine/execution_tree.h"

namespace ours {
namespace dataset {
WikiTextOp::WikiTextOp(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
                       std::unique_ptr<DataSchema> schema, const std::vector<std::string> &file_list,
                       int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id)
    : TextFileOp(num_workers, total_rows, worker_connector_size, std::move(schema), file_list, op_connector_size,
                 shuffle_files, num_devices, device_id) {}

// A print method typically used for debugging.
void WikiTextOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nRow count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nWikiText files list:\n";
    for (size_t i = 0; i < text_files_list_.size(); ++i) {
      out << " " << text_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}
}  // namespace dataset
}  // namespace ours
