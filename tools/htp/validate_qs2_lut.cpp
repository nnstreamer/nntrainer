// SPDX-License-Identifier: Apache-2.0
/** @file validate_qs2_lut.cpp
 * @brief Verify deployed QS2 artifacts once and bind a receipt to local files.
 * Build with the target C++17 compiler and -Inntrainer/tensor/htp_backend.
 * Run on the target after copying both model and prepared files:
 *   validate_qs2_lut model.bin /absolute/path/prepared_experts
 * Re-run after replacing either. No model or artifact payload is modified.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_prepared_qs2.h"
#include <cstdio>
#include <dirent.h>
#include <iostream>

using namespace nntrainer;
static void put(std::vector<uint8_t> &out, uint64_t word, unsigned width = 8) {
  for (unsigned i = 0; i < width; ++i)
    out.push_back(static_cast<uint8_t>(word >> (8u * i)));
}
static void stamp(std::vector<uint8_t> &out, const PreparedQs2Stamp &s) {
  for (auto word : s.words())
    put(out, word);
}
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: validate_qs2_lut model.bin prepared_directory\n";
    return 2;
  }
  struct Source {
    int fd;
    ~Source() {
      if (fd >= 0)
        close(fd);
    }
  } source{open(argv[1], O_RDONLY | O_CLOEXEC)};
  std::string temporary;
  try {
    if (source.fd < 0)
      throw std::runtime_error("cannot open model");
    const PreparedQs2Stamp model(source.fd);
    std::vector<std::string> files;
    DIR *dir = opendir(argv[2]);
    if (!dir)
      throw std::runtime_error("cannot open artifact directory");
    while (const auto *entry = readdir(dir)) {
      const std::string name(entry->d_name);
      if (name.size() > 7 && name.compare(name.size() - 7, 7, ".nntri2") == 0)
        files.push_back(std::string(argv[2]) + "/" + name);
    }
    closedir(dir);
    if (files.empty() || files.size() > 100000u)
      throw std::runtime_error("invalid artifact count");
    std::sort(files.begin(), files.end());
    std::vector<uint8_t> receipt{'N', 'N', 'T', 'R', 'V', '2', '0', '1'},
      payload;
    put(receipt, files.size());
    stamp(receipt, model);
    std::unordered_map<uint64_t, bool> offsets;
    for (const auto &path : files) {
      Source artifact{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
      if (artifact.fd < 0)
        throw std::runtime_error("cannot open artifact: " + path);
      const PreparedQs2Stamp before(artifact.fd);
      uint8_t header[64];
      preparedQs2Read(artifact.fd, header, sizeof(header), 0);
      if (!offsets.emplace(preparedFcU64(header + 16), true).second)
        throw std::runtime_error(
          "duplicate source offset in artifact directory");
      readPreparedQs2Cached(
        path, source.fd, preparedFcU64(header + 16), preparedFcU32(header + 40),
        preparedFcU32(header + 44), payload, nullptr, false);
      if (!(before == PreparedQs2Stamp(artifact.fd)))
        throw std::runtime_error("artifact changed during validation");
      receipt.insert(receipt.end(), header, header + 64);
      stamp(receipt, before);
    }
    if (!(model == PreparedQs2Stamp(source.fd)))
      throw std::runtime_error("model changed during validation");
    put(receipt, preparedQs2Crc(receipt.data(), receipt.size()), 4);
    const std::string dest = std::string(argv[2]) + "/.nntri2.validated";
    temporary = dest + "." + std::to_string(getpid()) + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char *>(receipt.data()),
                   receipt.size()))
      throw std::runtime_error("cannot write receipt");
    out.close();
    if (!out || std::rename(temporary.c_str(), dest.c_str()))
      throw std::runtime_error("cannot publish receipt");
    temporary.clear();
    std::cout << "verified " << files.size() << " artifacts; receipt " << dest
              << '\n';
  } catch (const std::exception &e) {
    if (!temporary.empty())
      std::remove(temporary.c_str());
    std::cerr << e.what() << '\n';
    return 1;
  }
}
