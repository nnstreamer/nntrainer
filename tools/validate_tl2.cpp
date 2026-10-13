// SPDX-License-Identifier: Apache-2.0
/** @file validate_tl2.cpp
 * @brief Validate every source and payload CRC once on the deployment device.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_prepared_tl2.h"
#include <iostream>
#if defined(__ANDROID__) || defined(__linux__)
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: validate_tl2 MODEL PREPARED_TL2\n";
    return 2;
  }
  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return 1;
  try {
    nntrainer::PreparedTl2 reader(argv[2], fd, false);
    const size_t count =
      reader.validate(fd, std::string(argv[2]) + ".validated");
    std::cout << "Validated " << count
              << " source weights and both payloads; receipt written\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    close(fd);
    return 1;
  }
  close(fd);
}

#else
int main() {
  std::cerr << "This deployment tool requires Linux or Android\n";
  return 2;
}
#endif
