#include "miniort/tools/gpt2_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace miniort {

namespace {

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::ostringstream oss;
  oss << input.rdbuf();
  return oss.str();
}

std::string EncodeUtf8(char32_t codepoint) {
  std::string out;
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
  return out;
}

std::vector<std::string> SplitUtf8Codepoints(const std::string& text) {
  std::vector<std::string> codepoints;
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    std::size_t length = 1;
    if ((lead & 0x80U) == 0) {
      length = 1;
    } else if ((lead & 0xE0U) == 0xC0U) {
      length = 2;
    } else if ((lead & 0xF0U) == 0xE0U) {
      length = 3;
    } else if ((lead & 0xF8U) == 0xF0U) {
      length = 4;
    }
    if (i + length > text.size()) {
      throw std::runtime_error("invalid UTF-8 sequence");
    }
    codepoints.push_back(text.substr(i, length));
    i += length;
  }
  return codepoints;
}

bool IsAsciiLetter(unsigned char c) {
  return std::isalpha(c) != 0;
}

bool IsAsciiDigit(unsigned char c) {
  return std::isdigit(c) != 0;
}

bool IsWhitespace(unsigned char c) {
  return std::isspace(c) != 0;
}

bool IsLetterLike(unsigned char c) {
  return IsAsciiLetter(c) || c >= 128;
}

std::uint32_t ParseHex4(std::string_view text) {
  if (text.size() != 4) {
    throw std::runtime_error("expected 4 hex digits");
  }
  std::uint32_t value = 0;
  for (char ch : text) {
    value <<= 4U;
    if (ch >= '0' && ch <= '9') {
      value += static_cast<std::uint32_t>(ch - '0');
    } else if (ch >= 'a' && ch <= 'f') {
      value += static_cast<std::uint32_t>(10 + ch - 'a');
    } else if (ch >= 'A' && ch <= 'F') {
      value += static_cast<std::uint32_t>(10 + ch - 'A');
    } else {
      throw std::runtime_error("invalid hex digit in JSON escape");
    }
  }
  return value;
}

std::string ParseJsonString(const std::string& text, std::size_t& pos) {
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  ++pos;
  std::string out;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (ch == '"') {
      return out;
    }
    if (ch != '\\') {
      out.push_back(ch);
      continue;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("unterminated JSON escape");
    }
    const char escaped = text[pos++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        out.push_back(escaped);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        if (pos + 4 > text.size()) {
          throw std::runtime_error("truncated JSON unicode escape");
        }
        std::uint32_t codepoint = ParseHex4(std::string_view(text).substr(pos, 4));
        pos += 4;
        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
          if (pos + 6 > text.size() || text[pos] != '\\' || text[pos + 1] != 'u') {
            throw std::runtime_error("missing low surrogate in JSON escape");
          }
          pos += 2;
          const std::uint32_t low = ParseHex4(std::string_view(text).substr(pos, 4));
          pos += 4;
          if (low < 0xDC00 || low > 0xDFFF) {
            throw std::runtime_error("invalid low surrogate in JSON escape");
          }
          codepoint = 0x10000 + (((codepoint - 0xD800) << 10U) | (low - 0xDC00));
        }
        out += EncodeUtf8(static_cast<char32_t>(codepoint));
        break;
      }
      default:
        throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

void SkipJsonWhitespace(const std::string& text, std::size_t& pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
}

std::unordered_map<std::string, std::int64_t> ParseVocabJson(const std::string& text) {
  std::unordered_map<std::string, std::int64_t> vocab;
  std::size_t pos = 0;
  SkipJsonWhitespace(text, pos);
  if (pos >= text.size() || text[pos] != '{') {
    throw std::runtime_error("vocab.json must start with an object");
  }
  ++pos;

  while (true) {
    SkipJsonWhitespace(text, pos);
    if (pos >= text.size()) {
      throw std::runtime_error("unterminated vocab.json object");
    }
    if (text[pos] == '}') {
      ++pos;
      break;
    }
    const std::string key = ParseJsonString(text, pos);
    SkipJsonWhitespace(text, pos);
    if (pos >= text.size() || text[pos] != ':') {
      throw std::runtime_error("expected ':' in vocab.json");
    }
    ++pos;
    SkipJsonWhitespace(text, pos);
    std::size_t value_start = pos;
    if (pos < text.size() && (text[pos] == '-' || std::isdigit(static_cast<unsigned char>(text[pos])) != 0)) {
      ++pos;
      while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
      }
    } else {
      throw std::runtime_error("expected integer token id in vocab.json");
    }
    const auto value = std::stoll(text.substr(value_start, pos - value_start));
    vocab.emplace(key, value);
    SkipJsonWhitespace(text, pos);
    if (pos >= text.size()) {
      throw std::runtime_error("unterminated vocab.json object");
    }
    if (text[pos] == ',') {
      ++pos;
      continue;
    }
    if (text[pos] == '}') {
      ++pos;
      break;
    }
    throw std::runtime_error("expected ',' or '}' in vocab.json");
  }

  return vocab;
}

std::unordered_map<std::uint8_t, std::string> BuildByteEncoder() {
  std::vector<int> values;
  for (int c = static_cast<int>('!'); c <= static_cast<int>('~'); ++c) {
    values.push_back(c);
  }
  for (int c = 0xA1; c <= 0xAC; ++c) {
    values.push_back(c);
  }
  for (int c = 0xAE; c <= 0xFF; ++c) {
    values.push_back(c);
  }

  std::vector<int> extra_values = values;
  int next = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(values.begin(), values.end(), b) == values.end()) {
      values.push_back(b);
      extra_values.push_back(256 + next);
      ++next;
    }
  }

  std::unordered_map<std::uint8_t, std::string> encoder;
  for (std::size_t i = 0; i < values.size(); ++i) {
    encoder[static_cast<std::uint8_t>(values[i])] = EncodeUtf8(static_cast<char32_t>(extra_values[i]));
  }
  return encoder;
}

std::unordered_map<std::string, std::size_t> ParseMerges(const std::string& text) {
  std::unordered_map<std::string, std::size_t> ranks;
  std::istringstream input(text);
  std::string line;
  std::size_t rank = 0;
  bool first_line = true;
  while (std::getline(input, line)) {
    if (first_line) {
      first_line = false;
      continue;
    }
    if (line.empty()) {
      continue;
    }
    std::istringstream pair_stream(line);
    std::string left;
    std::string right;
    if (!(pair_stream >> left >> right)) {
      continue;
    }
    ranks.emplace(left + "\n" + right, rank++);
  }
  return ranks;
}

}  // namespace

Gpt2Tokenizer::Gpt2Tokenizer(const std::filesystem::path& model_dir)
    : byte_encoder_(BuildByteEncoder()) {
  for (const auto& [byte_value, encoded] : byte_encoder_) {
    byte_decoder_[encoded] = byte_value;
  }

  vocab_ = ParseVocabJson(ReadFile(model_dir / "vocab.json"));
  merge_ranks_ = ParseMerges(ReadFile(model_dir / "merges.txt"));
  id_to_token_.resize(vocab_.size());
  for (const auto& [token, id] : vocab_) {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
      throw std::runtime_error("token id out of range in vocab.json");
    }
    id_to_token_[static_cast<std::size_t>(id)] = token;
  }
}

std::vector<std::string> Gpt2Tokenizer::SplitText(const std::string& text) const {
  std::vector<std::string> pieces;
  std::size_t i = 0;
  while (i < text.size()) {
    bool leading_space = false;
    if (text[i] == ' ' && i + 1 < text.size() && !IsWhitespace(static_cast<unsigned char>(text[i + 1]))) {
      leading_space = true;
      ++i;
    } else if (IsWhitespace(static_cast<unsigned char>(text[i]))) {
      const std::size_t start = i;
      while (i < text.size() && IsWhitespace(static_cast<unsigned char>(text[i]))) {
        ++i;
      }
      pieces.push_back(text.substr(start, i - start));
      continue;
    }

    const std::size_t start = i;
    if (text[i] == '\'') {
      constexpr std::string_view contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
      bool matched = false;
      for (const auto contraction : contractions) {
        if (text.compare(i, contraction.size(), contraction) == 0) {
          i += contraction.size();
          matched = true;
          break;
        }
      }
      if (!matched) {
        ++i;
      }
    } else if (IsLetterLike(static_cast<unsigned char>(text[i]))) {
      while (i < text.size() && IsLetterLike(static_cast<unsigned char>(text[i]))) {
        ++i;
      }
    } else if (IsAsciiDigit(static_cast<unsigned char>(text[i]))) {
      while (i < text.size() && IsAsciiDigit(static_cast<unsigned char>(text[i]))) {
        ++i;
      }
    } else {
      while (i < text.size()) {
        const auto ch = static_cast<unsigned char>(text[i]);
        if (IsWhitespace(ch) || IsLetterLike(ch) || IsAsciiDigit(ch)) {
          break;
        }
        if (text[i] == '\'' && i + 1 < text.size() && IsAsciiLetter(static_cast<unsigned char>(text[i + 1]))) {
          break;
        }
        ++i;
      }
    }

    std::string piece = text.substr(start, i - start);
    if (leading_space) {
      piece.insert(piece.begin(), ' ');
    }
    pieces.push_back(std::move(piece));
  }
  return pieces;
}

std::string Gpt2Tokenizer::EncodeBytes(const std::string& piece) const {
  std::string encoded;
  for (unsigned char byte_value : piece) {
    encoded += byte_encoder_.at(byte_value);
  }
  return encoded;
}

std::vector<std::string> Gpt2Tokenizer::ApplyBpe(const std::string& piece) const {
  if (const auto cache_it = bpe_cache_.find(piece); cache_it != bpe_cache_.end()) {
    return cache_it->second;
  }

  std::vector<std::string> symbols = SplitUtf8Codepoints(piece);
  if (symbols.empty()) {
    return {};
  }

  while (symbols.size() > 1) {
    std::size_t best_index = 0;
    std::size_t best_rank = std::numeric_limits<std::size_t>::max();
    bool found = false;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = merge_ranks_.find(symbols[i] + "\n" + symbols[i + 1]);
      if (it == merge_ranks_.end()) {
        continue;
      }
      if (it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
        found = true;
      }
    }
    if (!found) {
      break;
    }
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
  }

  bpe_cache_[piece] = symbols;
  return symbols;
}

std::vector<std::int64_t> Gpt2Tokenizer::Encode(const std::string& text) const {
  std::vector<std::int64_t> token_ids;
  for (const auto& piece : SplitText(text)) {
    const auto byte_encoded = EncodeBytes(piece);
    for (const auto& bpe_piece : ApplyBpe(byte_encoded)) {
      const auto it = vocab_.find(bpe_piece);
      if (it == vocab_.end()) {
        throw std::runtime_error("tokenizer vocab miss for piece: " + bpe_piece);
      }
      token_ids.push_back(it->second);
    }
  }
  return token_ids;
}

std::string Gpt2Tokenizer::Decode(const std::vector<std::int64_t>& token_ids) const {
  std::string merged;
  for (const auto token_id : token_ids) {
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
      throw std::runtime_error("token id out of range during decode");
    }
    merged += id_to_token_[static_cast<std::size_t>(token_id)];
  }

  std::string decoded;
  for (const auto& codepoint : SplitUtf8Codepoints(merged)) {
    const auto it = byte_decoder_.find(codepoint);
    if (it == byte_decoder_.end()) {
      throw std::runtime_error("tokenizer decode miss for codepoint");
    }
    decoded.push_back(static_cast<char>(it->second));
  }
  return decoded;
}

}  // namespace miniort
