// SPDX-License-Identifier: Apache-2.0
/** @brief Direct weight I/O: unaligned data, EOF, streams, concurrency, fd
 * reuse.
 * @file weight_direct_file_check.cpp
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "weight_direct_file.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

using namespace nntrainer;

int main(int argc, char **argv) {
#if defined(__linux__) || defined(__ANDROID__)
  const std::string root = argc > 1 ? argv[1] : "/tmp";
  auto create = [&](const char *suffix, uint8_t seed) {
    std::string path = root + suffix;
    std::vector<char> name(path.begin(), path.end());
    name.push_back(0);
    int fd = mkstemp(name.data());
    assert(fd >= 0);
    unlink(name.data());
    std::vector<uint8_t> bytes(3 * 1024 * 1024 + 137);
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<uint8_t>((i * 17 + i / 251) ^ seed);
    assert(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()));
    assert(fsync(fd) == 0);
    return std::make_pair(fd, bytes);
  };
  auto a = create("/nntr-direct-a-XXXXXX", 13);
  WeightDirectFiles files;
  auto check = [&](int fd, const std::vector<uint8_t> &data) {
    for (size_t at : {size_t(0), size_t(37), size_t(4095), size_t(1000003)}) {
      for (size_t n : {size_t(1), size_t(4), size_t(8193), size_t(1100007)}) {
        std::vector<uint8_t> out(n + 2, 0xee);
        assert(files.read(fd, out.data() + 1, n, at) == 0);
        assert(out.front() == 0xee && out.back() == 0xee);
        assert(std::memcmp(out.data() + 1, data.data() + at, n) == 0);
      }
    }
    uint8_t tail[137];
    assert(files.read(fd, tail, sizeof(tail), data.size() - sizeof(tail)) == 0);
    assert(std::memcmp(tail, data.data() + data.size() - sizeof(tail),
                       sizeof(tail)) == 0);
    assert(files.read(fd, tail, 1, data.size()) == -1);
    assert(files.read(fd, tail, 0, data.size()) == 0);
    assert(files.read(fd, tail, 1, UINT64_MAX) == -1);
    assert(files.read(fd, nullptr, 1, 0) == EINVAL);
    WeightDirectStream input(fd);
    auto &in = input.stream();
    in.seekg(37);
    in.read(reinterpret_cast<char *>(tail), 137);
    assert(in.good() && in.tellg() == 174);
    assert(std::memcmp(tail, data.data() + 37, 137) == 0);
    in.seekg(-137, std::ios::end);
    in.read(reinterpret_cast<char *>(tail), 137);
    assert(in.good() && in.tellg() == std::streamoff(data.size()));
    in.seekg(-1, std::ios::cur);
    in.read(reinterpret_cast<char *>(tail), 1);
    assert(in.good() && tail[0] == data.back());
    in.read(reinterpret_cast<char *>(tail), 1);
    assert(!in.good());
    in.clear();
    in.seekg(std::numeric_limits<std::streamoff>::min(), std::ios::cur);
    assert(!in.good());
  };
  std::vector<std::thread> workers;
  for (int i = 0; i < 6; ++i)
    workers.emplace_back([&] { check(a.first, a.second); });
  for (auto &worker : workers)
    worker.join();
  const int reused = a.first;
  close(a.first);
  auto b = create("/nntr-direct-b-XXXXXX", 29);
  assert(b.first == reused);
  check(b.first, b.second);
  close(b.first);
  std::cout << "PASS: direct reads, tails, streams, six readers and fd reuse\n";
#else
  (void)argc;
  (void)argv;
  std::cout << "SKIP: direct I/O requires Linux/Android\n";
#endif
}
