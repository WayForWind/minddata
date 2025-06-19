

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_GLOVE_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_GLOVE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/iterator.h"
#include "OURSdata/dataset/text/vectors.h"
#include "OURSdata/dataset/util/path.h"

namespace ours {
namespace dataset {
/// \brief Pre-train word vectors.
class GloVe : public Vectors {
 public:
  /// Constructor.
  GloVe() = default;

  /// Constructor.
  /// \param[in] map A map between string and vector.
  /// \param[in] dim Dimension of the vectors.
  GloVe(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim);

  /// Destructor.
  ~GloVe() = default;

  /// \brief Build Vectors from reading a pre-train vector file.
  /// \param[out] glove GloVe object which contains the pre-train vectors.
  /// \param[in] path Path to the pre-trained word vector file.
  /// \param[in] max_vectors This can be used to limit the number of pre-trained vectors loaded (default=0, no limit).
  static Status BuildFromFile(std::shared_ptr<GloVe> *glove, const std::string &path, int32_t max_vectors = 0);
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_GLOVE_H_
