
#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_TENSOR_HELPERS_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_TENSOR_HELPERS_H_

#include <memory>
#include <vector>

#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/include/dataset/constants.h"

namespace ours {
namespace dataset {
/// Recursive helper function to generate indices based on vector of SliceOptions. It recursively iterates through each
/// range represented by slice_options to generate a list of indices to be sliced.
/// \param[out] matrix Generated nested vector of indices
///       Example: For a 4 x 2 tensor, and with slice_list = {SliceOption({0})} (the first row), matrix will become
///       {{0}}. For slice_list = {SliceOption(all), SliceOption({0})} (the first column), matrix will become
///       {{0, 0}, {1, 0}, {2, 0}, {3, 0}}.
///       For slice_list = {SliceOption({0, 2})}, matrix will become {{0}, {2}}. The size of each nested array is always
///       equal to (slice_list).size().
/// \param[in] depth used to keep track of recursion level
/// \param[in] numbers vector used to represent current index
/// \param[in] matrix 2D vector to be populated with desired indices
/// \param[in] slice_options vector of SliceOption objects
void IndexGeneratorHelper(int8_t depth, std::vector<dsize_t> *numbers, const std::vector<SliceOption> &slice_list,
                          std::vector<std::vector<dsize_t>> *matrix);

/// Generate indices based on vector of SliceOptions
/// Calls the recursive helper function IndexGeneratorHelper
/// \param[in] slice_list vector of SliceOption objects. Note: If the user passes
///       {SliceOption(true), SliceOption(true)}, it will return a M x 2 vector, instead of reducing it to
///       {SliceOption(true)} first to only generate a M x 1 vector.
/// \return std::vector<std::vector<dsize_t>> 2D vector of generated indices, M x (slice_list).size()
std::vector<std::vector<dsize_t>> IndexGenerator(const std::vector<SliceOption> &slice_list);
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_TENSOR_HELPERS_H_
