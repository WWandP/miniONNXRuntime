#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace miniort {

class Gpt2Tokenizer {
 public:
  explicit Gpt2Tokenizer(const std::filesystem::path& model_dir);

  std::vector<std::int64_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<std::int64_t>& token_ids) const;

 private:
  std::vector<std::string> SplitText(const std::string& text) const;
  std::string EncodeBytes(const std::string& piece) const;
  std::vector<std::string> ApplyBpe(const std::string& piece) const;

  std::unordered_map<std::string, std::int64_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, std::size_t> merge_ranks_;
  mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
  std::unordered_map<std::uint8_t, std::string> byte_encoder_;
  std::unordered_map<std::string, std::uint8_t> byte_decoder_;
};

}  // namespace miniort
