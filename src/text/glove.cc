

#include "OURSdata/dataset/text/glove.h"

#include "utils/file_utils.h"

namespace ours {
namespace dataset {
GloVe::GloVe(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim) : Vectors(map, dim) {}

Status CheckGloVe(const std::string &file_path) {
  Path path = Path(file_path);
  if (path.Exists() && !path.IsDirectory()) {
    std::string basename = path.Basename();
    size_t dot = basename.rfind('.');
    std::string suffix = basename.substr(dot + 1);
    std::string sub_name = basename.substr(0, dot);
    dot = sub_name.rfind('.');
    std::string glove_name = sub_name.substr(0, dot);
    dot = glove_name.rfind('.');
    std::string infix = glove_name.substr(dot + 1);
    std::string prefix = glove_name.substr(0, dot);
    if (suffix != "txt" || infix != "6B" || prefix != "glove") {
      RETURN_STATUS_UNEXPECTED("GloVe: invalid file, can not find file 'glove.6B.*.txt', but got: " + file_path);
    }
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("GloVe: invalid file, failed to open GloVe file.");
  }
}

Status GloVe::BuildFromFile(std::shared_ptr<GloVe> *glove, const std::string &path, int32_t max_vectors) {
  RETURN_UNEXPECTED_IF_NULL(glove);
  RETURN_IF_NOT_OK(CheckGloVe(path));
  std::unordered_map<std::string, std::vector<float>> map;
  int vector_dim = -1;
  RETURN_IF_NOT_OK(Load(path, max_vectors, &map, &vector_dim));
  *glove = std::make_shared<GloVe>(std::move(map), vector_dim);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
