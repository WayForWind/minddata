

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUILD_SENTENCE_PIECE_VOCAB_NODE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUILD_SENTENCE_PIECE_VOCAB_NODE_H_

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"
#include "OURSdata/dataset/include/dataset/datasets.h"

namespace ours {
namespace dataset {
class BuildSentenceVocabNode : public DatasetNode {
 public:
  /// \brief Constructor
  BuildSentenceVocabNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<SentencePieceVocab> vocab,
                         const std::vector<std::string> &col_names, int32_t vocab_size, float character_coverage,
                         SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params);

  /// \brief Destructor
  ~BuildSentenceVocabNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kBuildSentencePieceVocabNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;

  /// \brief Getter functions
  const std::shared_ptr<SentencePieceVocab> &GetVocab() const { return vocab_; }
  const std::vector<std::string> &ColNames() const { return col_names_; }
  int32_t VocabSize() const { return vocab_size_; }
  float CharacterCoverage() const { return character_coverage_; }
  SentencePieceModel ModelType() const { return model_type_; }
  const std::unordered_map<std::string, std::string> &Params() const { return params_; }

 private:
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::vector<std::string> col_names_;
  int32_t vocab_size_;
  float character_coverage_;
  SentencePieceModel model_type_;
  std::unordered_map<std::string, std::string> params_;
};
}  // namespace dataset
}  // namespace ours
#endif  // #ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUILD_SENTENCE_PIECE_VOCAB_NODE_H_
