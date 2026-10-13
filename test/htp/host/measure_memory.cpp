// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file measure_memory.cpp
 * @brief Linux/Android command memory sampler, including DMA and model cache.
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <set>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
struct Sample {
  size_t anon = 0, file = 0, shmem = 0, swap = 0, dma = 0, cache = 0;
};

Sample sampleProcess(pid_t pid) {
  Sample s;
  const std::string proc = "/proc/" + std::to_string(pid);
  char line[1024];
  FILE *status = std::fopen((proc + "/status").c_str(), "r");
  if (status) {
    while (std::fgets(line, sizeof(line), status)) {
      std::sscanf(line, "RssAnon: %zu", &s.anon);
      std::sscanf(line, "RssFile: %zu", &s.file);
      std::sscanf(line, "RssShmem: %zu", &s.shmem);
      std::sscanf(line, "VmSwap: %zu", &s.swap);
    }
    std::fclose(status);
  }
  // FastRPC mappings need not appear in RSS. Count each exported allocation
  // once, even when several file descriptors refer to the same DMA buffer.
  std::set<unsigned long long> seen;
  const std::string fdinfo = proc + "/fdinfo/";
  DIR *dir = opendir(fdinfo.c_str());
  if (dir) {
    while (auto *entry = readdir(dir)) {
      if (entry->d_name[0] < '0' || entry->d_name[0] > '9')
        continue;
      FILE *f = std::fopen((fdinfo + entry->d_name).c_str(), "r");
      if (!f)
        continue; // A descriptor can close between readdir and fopen.
      size_t size = 0;
      unsigned long long inode = 0;
      bool exported = false;
      while (std::fgets(line, sizeof(line), f)) {
        std::sscanf(line, "size: %zu", &size);
        std::sscanf(line, "ino: %llu", &inode);
        exported |= std::strncmp(line, "exp_name:", 9) == 0;
      }
      if (exported && size && seen.insert(inode).second)
        s.dma += size;
      std::fclose(f);
    }
    closedir(dir);
  }
  return s;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s OUTPUT.csv MODEL.bin COMMAND [ARGS...]\n",
                 argv[0]);
    return 2;
  }
  const int fd = open(argv[2], O_RDONLY | O_CLOEXEC);
  struct stat st {};
  if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) {
    std::perror("model file");
    if (fd >= 0)
      close(fd);
    return 2;
  }
  const long page = sysconf(_SC_PAGESIZE);
  if (page <= 0) {
    close(fd);
    return 2;
  }
  // mincore observes residency without faulting the model's pages into RAM.
  const size_t size = static_cast<size_t>(st.st_size);
  void *mapping = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) {
    std::perror("model mmap");
    close(fd);
    return 2;
  }
  FILE *csv = std::fopen(argv[1], "w");
  if (!csv) {
    std::perror("output CSV");
    munmap(mapping, size);
    close(fd);
    return 2;
  }
  fcntl(fileno(csv), F_SETFD, FD_CLOEXEC);
  const pid_t pid = fork();
  if (pid == 0) {
    execvp(argv[3], argv + 3);
    std::perror("execvp");
    _exit(127);
  }
  if (pid < 0) {
    std::perror("fork");
    std::fclose(csv);
    munmap(mapping, size);
    close(fd);
    return 2;
  }
  std::vector<unsigned char> resident((size + page - 1) / page);
  std::fprintf(csv, "ms,rss_anon_kb,rss_file_kb,rss_shmem_kb,swap_kb,"
                    "dma_bytes,model_cache_bytes\n");
  const auto start = std::chrono::steady_clock::now();
  int status = 0;
  bool valid = true;
  size_t peak = 0;
  for (;;) {
    const pid_t ended = waitpid(pid, &status, WNOHANG);
    if (ended == pid)
      break;
    if (ended < 0) {
      if (errno == EINTR)
        continue;
      std::perror("waitpid");
      valid = false;
      break;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    Sample s = sampleProcess(pid);
    if (mincore(mapping, size, resident.data())) {
      std::perror("model mincore");
      valid = false;
    } else {
      for (auto value : resident)
        s.cache += (value & 1) * page;
    }
    // Conservative accounting: include swapped-out anonymous memory too.
    // A legacy loader that mmaps the model can double-count its file RSS;
    // the bounded/direct reader does not map it into the measured process.
    peak = std::max(peak, (s.anon + s.file + s.shmem + s.swap) * 1024 + s.dma +
                            s.cache);
    std::fprintf(csv, "%.1f,%zu,%zu,%zu,%zu,%zu,%zu\n", ms, s.anon, s.file,
                 s.shmem, s.swap, s.dma, s.cache);
    std::fflush(csv);
    usleep(50000);
  }
  std::fprintf(stderr, "[memory] sampled peak including swap/cache: %.3f GB\n",
               static_cast<double>(peak) / 1e9);
  std::fclose(csv);
  munmap(mapping, size);
  close(fd);
  if (!valid)
    return 2;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
