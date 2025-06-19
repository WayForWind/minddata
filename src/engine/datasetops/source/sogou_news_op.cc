

#include "OURSdata/dataset/engine/datasetops/source/sogou_news_op.h"

#include <utility>

#include "include/common/debug/common.h"

namespace ours {
namespace dataset {
SogouNewsOp::SogouNewsOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size,
                         int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id,
                         char field_delim, const std::vector<std::shared_ptr<BaseRecord>> &column_default,
                         const std::vector<std::string> &column_name,
                         const std::vector<std::string> &sogou_news_files_list)
    : CsvOp(sogou_news_files_list, field_delim, column_default, column_name, num_workers, num_samples,
            worker_connector_size, op_connector_size, shuffle_files, num_devices, device_id) {}

void SogouNewsOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nSogouNews files list:\n";
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      out << " " << csv_files_list_[i];
    }
    out << "\n\n";
  }
}
}  // namespace dataset
}  // namespace ours
