

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_VECTORS_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_VECTORS_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/iterator.h"

namespace ours {
namespace dataset {
/// \brief Pre-train word vectors.
class Vectors {
 public:
  /// Constructor.
  Vectors() = default;

  /// Constructor.
  /// \param[in] map A map between string and vector.
  /// \param[in] dim Dimension of the vectors.
  Vectors(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim);

  /// Destructor.
  virtual ~Vectors() = default;

  /// \brief Build Vectors from reading a pre-train vector file.
  /// \param[out] vectors Vectors object which contains the pre-train vectors.
  /// \param[in] path Path to the pre-trained word vector file.
  /// \param[in] max_vectors This can be used to limit the number of pre-trained vectors loaded (default=0, no limit).
  static Status BuildFromFile(std::shared_ptr<Vectors> *vectors, const std::string &path, int32_t max_vectors = 0);

  /// \brief Look up embedding vectors of token.
  /// \param[in] token A token to be looked up.
  /// \param[in] unk_init In case of the token is out-of-vectors (OOV), the result will be initialized with `unk_init`.
  ///     (default={}, means to initialize with zero vectors).
  /// \param[in] lower_case_backup Whether to look up the token in the lower case (Default = false).
  /// \return The vector of the input token.
  virtual std::vector<float> Lookup(const std::string &token, const std::vector<float> &unk_init = {},
                                    bool lower_case_backup = false);

  /// \brief Getter of dimension.
  const int32_t &Dim() const { return dim_; }

 protected:
  /// \brief Infer the shape of the pre-trained word vector file.
  /// \param[in] path Path to the pre-trained word vector file.
  /// \param[in] max_vectors Maximum number of pre-trained word vectors to be read.
  /// \param[out] num_lines The number of lines of the file.
  /// \param[out] header_num_lines The number of lines of file header.
  /// \param[out] vector_dim The dimension of the vectors in the file.
  static Status InferShape(const std::string &path, int32_t max_vectors, int32_t *num_lines, int32_t *header_num_lines,
                           int32_t *vector_dim);

  /// \brief Load map from reading a pre-train vector file.
  /// \param[in] path Path to the pre-trained word vector file.
  /// \param[in] max_vectors This can be used to limit the number of pre-trained vectors loaded, must be non negative.
  /// \param[out] map The map between words and vectors.
  /// \param[out] vector_dim The dimension of the vectors in the file.
  static Status Load(const std::string &path, int32_t max_vectors,
                     std::unordered_map<std::string, std::vector<float>> *map, int32_t *vector_dim);

  int32_t dim_;
  std::unordered_map<std::string, std::vector<float>> map_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_VECTORS_H_
