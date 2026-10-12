// SPDX-License-Identifier: Apache-2.0
/** @file prepared_qs2_check.cpp
 * @brief Cross-language offline encoder/reader and production repacker check.
 */
#include "htp_prepared_qs2.h"
#include "hvx_gemv_u8i2_lut.h"
#include <iostream>
#include <iterator>

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6)
    return 2;
  try {
    const auto K = static_cast<uint32_t>(std::stoul(argv[3]));
    const auto N = static_cast<uint32_t>(std::stoul(argv[4]));
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)), {});
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("open source failed");
    std::vector<uint8_t> got,
      arena(static_cast<size_t>(K) * N / 4u + 128u, 0xa5);
    nntrainer::readPreparedQs2Cached(argv[2], fd, 0, K, N, got, arena.data());
    std::vector<uint8_t> expected(source.size());
    hvx_i2_lut_repack(source.data(), N, N * 8u, expected.data(), N * 8u, 1024u,
                      0, K / 32u);
    std::memcpy(expected.data() + K * N / 4u, source.data() + K * N / 4u,
                source.size() - K * N / 4u);
    if (std::memcmp(got.data() + K * N / 4u, expected.data() + K * N / 4u,
                    got.size() - K * N / 4u))
      throw std::runtime_error("Python and C weight metadata differ");
    const size_t codes = static_cast<size_t>(K) * N / 4u;
    if (std::memcmp(arena.data(), expected.data(), codes))
      throw std::runtime_error("cold direct load differs");
    // Direct reads must populate the arena even when cached scratch holds
    // deliberately wrong codes. Guard the metadata/end of the destination.
    std::fill(arena.begin(), arena.end(), 0xa5);
    std::fill(got.begin(), got.begin() + codes, 0);
    nntrainer::readPreparedQs2Cached(argv[2], fd, 0, K, N, got, arena.data());
    if (std::memcmp(arena.data(), expected.data(), codes) ||
        std::memcmp(got.data() + codes, expected.data() + codes,
                    got.size() - codes))
      throw std::runtime_error("warm direct load differs");
    for (size_t i = codes; i < arena.size(); ++i)
      if (arena[i] != 0xa5)
        throw std::runtime_error("direct load overwrote guard");
    nntrainer::readPreparedQs2Cached(argv[2], fd, 0, K, N, got);
    if (got != expected)
      throw std::runtime_error("cached load differs");
    if (argc == 6 && (std::string(argv[5]) == "receipt" ||
                      std::string(argv[5]) == "no-receipt")) {
      const int artifact = open(argv[2], O_RDONLY);
      uint8_t header[64];
      nntrainer::preparedQs2Read(artifact, header, sizeof(header), 0);
      const bool match = nntrainer::preparedQs2ReceiptMatch(
        argv[2], nntrainer::PreparedQs2Stamp(fd),
        nntrainer::PreparedQs2Stamp(artifact), header);
      close(artifact);
      if (match != (std::string(argv[5]) == "receipt"))
        throw std::runtime_error("wrong receipt reuse decision");
    } else if (argc == 6) {
      const char *name = std::string(argv[5]) == "source" ? argv[1] : argv[2];
      const int edit = open(name, O_RDWR);
      unsigned char byte;
      if (edit < 0 || pread(edit, &byte, 1, 80) != 1)
        throw std::runtime_error("edit read failed");
      byte ^= 1;
      if (pwrite(edit, &byte, 1, 80) != 1)
        throw std::runtime_error("edit write failed");
      close(edit);
      bool rejected = false;
      try {
        nntrainer::readPreparedQs2Cached(argv[2], fd, 0, K, N, got,
                                         arena.data());
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("modified file reused validation");
    }
    close(fd);
    std::cout << "offline QS2: reader and production layout match\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
