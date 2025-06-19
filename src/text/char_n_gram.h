

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_CHAR_N_GRAM_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_CHAR_N_GRAM_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "OURSdata/dataset/text/vectors.h"

namespace ours {
namespace dataset {
/// \brief Build CharNGram vectors from reading a Pre-train word vectors.
class CharNGram : public Vectors {
 public:
  // Constructor.
  CharNGram() = default;

  /// Constructor.
  /// \param[in] map A map between string and vector.
  /// \param[in] dim Dimension of the vectors.
  CharNGram(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim);

  // Destructor.
  ~CharNGram() = default;

  /// \brief Build CharNGram from reading a CharNGram pre-train vector file.
  /// \param[out] char_n_gram CharNGram object which contains the pre-train vectors.
  /// \param[in] path Path to the CharNGram pre-trained word vector file.
  /// \param[in] max_vectors This can be used to limit the number of pre-trained vectors loaded (default=0, no limit).
  static Status BuildFromFile(std::shared_ptr<CharNGram> *char_n_gram, const std::string &path,
                              int32_t max_vectors = 0);

  /// \brief Look up embedding vectors of token.
  /// \param[in] token A token to be looked up.
  /// \param[in] unk_init In case of the token is out-of-vectors (OOV), the result will be initialized with `unk_init`.
  ///     (default={}, means to initialize with zero vectors).
  /// \param[in] lower_case_backup Whether to look up the token in the lower case (Default = false).
  /// \return The vector of the input token.
  std::vector<float> Lookup(const std::string &token, const std::vector<float> &unk_init = {},
                            bool lower_case_backup = false);
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_CHAR_N_GRAM_H_
