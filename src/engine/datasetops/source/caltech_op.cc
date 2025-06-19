
#include "OURSdata/dataset/engine/datasetops/source/caltech_op.h"

#include <map>
#include <memory>
#include <set>
#include <utility>

namespace ours {
namespace dataset {
const std::set<std::string> kExts = {".jpg", ".JPEG"};
const std::map<std::string, int32_t> kClassIndex = {};
CaltechOp::CaltechOp(int32_t num_workers, const std::string &file_dir, int32_t queue_size, bool do_decode,
                     std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : ImageFolderOp(num_workers, file_dir, queue_size, false, do_decode, kExts, kClassIndex, std::move(data_schema),
                    std::move(sampler)) {}
}  // namespace dataset
}  // namespace ours
