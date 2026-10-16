// SPDX-License-Identifier: Apache-2.0
// External, sampled process-memory accounting for Linux/Android benchmarks.
// This is a diagnostic harness, not an allocator or a device-wide RSS meter.

#include <cstdio>
#if defined(__linux__)

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
using U64 = uint64_t;

static volatile sig_atomic_t child_pid = 0;
static volatile sig_atomic_t forwarded_signal = 0;

static void forwardSignal(int signum) {
  const int saved_errno = errno;
  forwarded_signal = signum;
  const pid_t pid = static_cast<pid_t>(child_pid);
  if (pid > 0 && kill(-pid, signum) != 0)
    kill(pid, signum);
  errno = saved_errno;
}

static std::string trim(const std::string &s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return {};
  return s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}

static bool parseUnsigned(const std::string &s, U64 &value,
                          const char *suffix = "") {
  const std::string text = trim(s);
  if (text.empty() || text[0] < '0' || text[0] > '9')
    return false;
  errno = 0;
  char *end = nullptr;
  const unsigned long long result = std::strtoull(text.c_str(), &end, 10);
  if (errno == ERANGE || trim(end) != suffix)
    return false;
  value = static_cast<U64>(result);
  return true;
}

static U64 add(U64 a, U64 b, bool &overflow) {
  if (b > std::numeric_limits<U64>::max() - a) {
    overflow = true;
    return std::numeric_limits<U64>::max();
  }
  return a + b;
}

struct Sample {
  double begin_ms = 0, end_ms = 0;
  bool status_read = false, anon_present = false, rss_present = false;
  bool hwm_present = false, fdinfo_directory_read = false, overflow = false;
  U64 anon_kib = 0, rss_kib = 0, hwm_kib = 0;
  U64 fd_raw_bytes = 0, fd_dedup_bytes = 0, unknown_identity_bytes = 0;
  U64 numeric_fds = 0, fdinfo_read = 0, fdinfo_errors = 0;
  U64 size_fds = 0, known_identity_fds = 0, unique_known_buffers = 0;
  U64 duplicate_fds = 0, unknown_identity_fds = 0;
  U64 no_mount_identity_fds = 0, inode_size_conflicts = 0;
  U64 exporter_evidence_fds = 0, invalid_fields = 0;
  bool pre_exec = false;
  bool completeDedup() const {
    return fdinfo_directory_read && !fdinfo_errors && !unknown_identity_fds &&
           !no_mount_identity_fds && !inode_size_conflicts && !invalid_fields &&
           !overflow;
  }
  bool compositeAvailable() const {
    return status_read && anon_present && fdinfo_directory_read && !overflow &&
           anon_kib <= (std::numeric_limits<U64>::max() - fd_raw_bytes) / 1024;
  }
  U64 totalRaw() const { return anon_kib * 1024 + fd_raw_bytes; }
  U64 totalDedup() const { return anon_kib * 1024 + fd_dedup_bytes; }
};

static Sample readProc(const std::string &proc_dir) {
  Sample s;
  std::ifstream status(proc_dir + "/status");
  s.status_read = status.is_open();
  std::string line;
  while (std::getline(status, line)) {
    const auto sep = line.find(':');
    if (sep == std::string::npos)
      continue;
    const std::string key = trim(line.substr(0, sep));
    U64 value = 0;
    if (key == "RssAnon" || key == "VmRSS" || key == "VmHWM") {
      if (!parseUnsigned(line.substr(sep + 1), value, "kB")) {
        ++s.invalid_fields;
        continue;
      }
      if (key == "RssAnon") {
        s.anon_kib = value;
        s.anon_present = true;
      } else if (key == "VmRSS") {
        s.rss_kib = value;
        s.rss_present = true;
      } else {
        s.hwm_kib = value;
        s.hwm_present = true;
      }
    }
  }
  if (status.bad())
    s.status_read = false;

  const std::string fd_dir = proc_dir + "/fdinfo";
  DIR *dir = opendir(fd_dir.c_str());
  if (!dir)
    return s;
  s.fdinfo_directory_read = true;
  std::map<std::pair<std::string, U64>, U64> buffers;
  for (;;) {
    errno = 0;
    dirent *entry = readdir(dir);
    if (!entry) {
      if (errno)
        ++s.fdinfo_errors;
      break;
    }
    const std::string name(entry->d_name);
    if (name.empty() ||
        name.find_first_not_of("0123456789") != std::string::npos)
      continue;
    ++s.numeric_fds;
    std::ifstream file(fd_dir + "/" + name);
    if (!file.is_open()) {
      ++s.fdinfo_errors;
      continue;
    }
    bool has_size = false, has_inode = false, has_mount = false;
    bool exporter = false;
    U64 size = 0, inode = 0, mount = 0;
    while (std::getline(file, line)) {
      const auto sep = line.find(':');
      if (sep == std::string::npos)
        continue;
      const std::string key = trim(line.substr(0, sep));
      U64 value = 0;
      if (key == "size" || key == "ino" || key == "mnt_id") {
        if (!parseUnsigned(line.substr(sep + 1), value)) {
          ++s.invalid_fields;
          continue;
        }
        if (key == "size") {
          size = value;
          has_size = true;
        } else if (key == "ino") {
          inode = value;
          has_inode = value != 0;
        } else {
          mount = value;
          has_mount = true;
        }
      } else if (key == "exp_name") {
        exporter = !trim(line.substr(sep + 1)).empty();
      }
    }
    if (file.bad()) {
      ++s.fdinfo_errors;
      continue;
    }
    ++s.fdinfo_read;
    if (!has_size)
      continue;
    ++s.size_fds;
    s.exporter_evidence_fds += exporter;
    s.fd_raw_bytes = add(s.fd_raw_bytes, size, s.overflow);
    if (!has_inode) {
      ++s.unknown_identity_fds;
      s.unknown_identity_bytes =
        add(s.unknown_identity_bytes, size, s.overflow);
      continue;
    }
    ++s.known_identity_fds;
    s.no_mount_identity_fds += !has_mount;
    const auto key =
      std::make_pair(has_mount ? std::to_string(mount) : "?", inode);
    auto found = buffers.find(key);
    if (found == buffers.end()) {
      buffers.emplace(key, size);
    } else {
      ++s.duplicate_fds;
      if (found->second != size) {
        ++s.inode_size_conflicts;
        found->second = std::max(found->second, size);
      }
    }
  }
  closedir(dir);
  s.unique_known_buffers = buffers.size();
  s.fd_dedup_bytes = s.unknown_identity_bytes;
  for (const auto &buffer : buffers)
    s.fd_dedup_bytes = add(s.fd_dedup_bytes, buffer.second, s.overflow);
  return s;
}

static void quoted(std::ostream &out, const std::string &text) {
  out << '"';
  for (unsigned char c : text) {
    switch (c) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 32)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << unsigned(c) << std::dec << std::setfill(' ');
      else
        out << c;
    }
  }
  out << '"';
}

static void sampleJson(std::ostream &out, const Sample &s) {
  out << std::boolalpha << "{\"begin_ms\":" << s.begin_ms
      << ",\"end_ms\":" << s.end_ms << ",\"scan_ms\":" << s.end_ms - s.begin_ms
      << ",\"pre_exec\":" << s.pre_exec << ",\"status_read\":" << s.status_read
      << ",\"rss_anon_present\":" << s.anon_present
      << ",\"vmrss_present\":" << s.rss_present
      << ",\"vmhwm_present\":" << s.hwm_present
      << ",\"rss_anon_kib\":" << s.anon_kib << ",\"vmrss_kib\":" << s.rss_kib
      << ",\"vmhwm_kib\":" << s.hwm_kib
      << ",\"fdinfo_directory_read\":" << s.fdinfo_directory_read
      << ",\"fdinfo_size_bytes_raw\":" << s.fd_raw_bytes
      << ",\"fdinfo_size_bytes_dedup_or_raw_fallback\":" << s.fd_dedup_bytes
      << ",\"numeric_fds\":" << s.numeric_fds
      << ",\"fdinfo_read\":" << s.fdinfo_read
      << ",\"fdinfo_read_errors\":" << s.fdinfo_errors
      << ",\"size_fds\":" << s.size_fds
      << ",\"known_identity_fds\":" << s.known_identity_fds
      << ",\"unique_known_buffers\":" << s.unique_known_buffers
      << ",\"duplicate_fds\":" << s.duplicate_fds
      << ",\"unknown_identity_fds\":" << s.unknown_identity_fds
      << ",\"unknown_identity_bytes\":" << s.unknown_identity_bytes
      << ",\"ino_without_mount_fds\":" << s.no_mount_identity_fds
      << ",\"inode_size_conflicts\":" << s.inode_size_conflicts
      << ",\"fdinfo_exporter_evidence_fds\":" << s.exporter_evidence_fds
      << ",\"invalid_fields\":" << s.invalid_fields
      << ",\"counter_overflow\":" << s.overflow
      << ",\"dedup_complete_for_readable_size_fds\":" << s.completeDedup()
      << ",\"composite_available\":" << s.compositeAvailable()
      << ",\"anon_plus_fdinfo_raw_bytes\":";
  if (s.compositeAvailable())
    out << s.totalRaw();
  else
    out << "null";
  out << ",\"anon_plus_fdinfo_dedup_or_raw_fallback_bytes\":";
  if (s.compositeAvailable())
    out << s.totalDedup();
  else
    out << "null";
  out << '}';
}

static double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static void writeReport(const std::string &output, const std::string &temporary,
                        const std::vector<Sample> &samples,
                        const std::vector<std::string> &command, pid_t pid,
                        int period_ms, int status, bool waited, double wall_ms,
                        double started_epoch_ms, bool snapshot_mode,
                        long final_maxrss_kib = -1) {
  size_t dedup_index = 0, raw_index = 0, valid_samples = 0;
  U64 hwm = 0, rss = 0, anon = 0;
  bool have_hwm = false;
  double scan_sum = 0, scan_max = 0;
  std::vector<double> intervals;
  for (size_t i = 0; i < samples.size(); ++i) {
    const Sample &s = samples[i];
    if (s.compositeAvailable()) {
      if (!valid_samples || s.totalDedup() > samples[dedup_index].totalDedup())
        dedup_index = i;
      if (!valid_samples || s.totalRaw() > samples[raw_index].totalRaw())
        raw_index = i;
      ++valid_samples;
    }
    if (s.hwm_present) {
      hwm = std::max(hwm, s.hwm_kib);
      have_hwm = true;
    }
    if (s.rss_present)
      rss = std::max(rss, s.rss_kib);
    if (s.anon_present)
      anon = std::max(anon, s.anon_kib);
    scan_sum += s.end_ms - s.begin_ms;
    scan_max = std::max(scan_max, s.end_ms - s.begin_ms);
    if (i)
      intervals.push_back(s.begin_ms - samples[i - 1].begin_ms);
  }
  double interval_sum = 0;
  for (double x : intervals)
    interval_sum += x;
  std::sort(intervals.begin(), intervals.end());
  const auto percentile = [&](double p) {
    if (intervals.empty())
      return 0.0;
    return intervals[std::min(intervals.size() - 1,
                              static_cast<size_t>(p * (intervals.size() - 1)))];
  };
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("cannot write output temporary file");
  out << std::setprecision(12) << std::boolalpha;
  out << "{\n\"schema_version\":1,\n\"mode\":";
  quoted(out, snapshot_mode ? "fake-proc-or-snapshot" : "supervised-child");
  out << ",\n\"pid\":" << pid << ",\n\"command\":[";
  for (size_t i = 0; i < command.size(); ++i) {
    if (i)
      out << ',';
    quoted(out, command[i]);
  }
  out << "],\n\"started_epoch_ms\":" << started_epoch_ms
      << ",\n\"wall_ms\":" << wall_ms
      << ",\n\"requested_period_ms\":" << period_ms
      << ",\n\"sample_count\":" << samples.size()
      << ",\n\"composite_available_sample_count\":" << valid_samples
      << ",\n\"sample_interval_ms\":{\"count\":" << intervals.size()
      << ",\"min\":" << percentile(0) << ",\"mean\":"
      << (intervals.empty() ? 0 : interval_sum / intervals.size())
      << ",\"p50\":" << percentile(0.5) << ",\"p95\":" << percentile(0.95)
      << ",\"max\":" << percentile(1) << "}"
      << ",\n\"sample_scan_ms\":{\"mean\":"
      << (samples.empty() ? 0 : scan_sum / samples.size())
      << ",\"max\":" << scan_max << "}"
      << ",\n\"child_waited\":" << waited << ",\n\"child_exit_code\":";
  if (waited && WIFEXITED(status))
    out << WEXITSTATUS(status);
  else
    out << "null";
  out << ",\n\"child_signal\":";
  if (waited && WIFSIGNALED(status))
    out << WTERMSIG(status);
  else
    out << "null";
  out << ",\n\"forwarded_signal\":" << forwarded_signal
      << ",\n\"dma_buffer_identity_verified\":false"
      << ",\n\"dedup_identity\":\"mnt_id+ino; ino-only namespace when mnt_id "
         "absent; no ino counts raw per fd\""
      << ",\n\"peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes\":";
  if (valid_samples)
    out << samples[dedup_index].totalDedup();
  else
    out << "null";
  out << ",\n\"peak_anon_plus_fdinfo_raw_bytes\":";
  if (valid_samples)
    out << samples[raw_index].totalRaw();
  else
    out << "null";
  out << ",\n\"peak_dedup_sample\":";
  if (valid_samples)
    sampleJson(out, samples[dedup_index]);
  else
    out << "null";
  out << ",\n\"peak_raw_sample\":";
  if (valid_samples)
    sampleJson(out, samples[raw_index]);
  else
    out << "null";
  out << ",\n\"max_observed_vm_hwm_kib\":";
  if (have_hwm)
    out << hwm;
  else
    out << "null";
  out << ",\n\"child_final_ru_maxrss_kib\":";
  if (final_maxrss_kib >= 0)
    out << final_maxrss_kib;
  else
    out << "null";
  out << ",\n\"max_sampled_vm_rss_kib\":" << rss
      << ",\n\"max_sampled_rss_anon_kib\":" << anon << ",\n\"limitations\":[";
  const std::vector<std::string> limitations = {
    "Sampled process accounting from before exec through child exit; "
    "transients between samples can be missed.",
    "Only the supervised PID is read, not descendants or device-wide memory; "
    "the monitor's own memory is excluded.",
    "Status and fdinfo are read sequentially, not as an atomic snapshot; "
    "opening, closing and reusing fds can race the scan.",
    "A size: field is not verified as DMA memory. exp_name is evidence only; "
    "dma_buffer_identity_verified remains false.",
    "Missing inode identities are counted per fd without deduplication; "
    "missing mount IDs use an unknown-mount inode namespace.",
    "Conflicting sizes for one identity use the largest observed size and are "
    "flagged. Unreadable or malformed fdinfo is flagged.",
    "Allocated fd sizes are not necessarily resident bytes. Private DSP heap, "
    "driver allocations, VTCM and page cache may be absent.",
    "Host VmRSS/VmHWM and mapped HTP arena sizes must not be added to the "
    "composite; they can overlap fd-backed memory.",
    "VmHWM is the maximum observed kernel host high-water; a process exiting "
    "between polls may hide its final high-water.",
    "child_final_ru_maxrss_kib comes from wait4 on Linux/Android and captures "
    "the terminated child host RSS high-water; it must not be added to sampled "
    "fd sizes or the composite.",
    "Sampling adds CPU and procfs activity; compare baseline and candidate "
    "with the same monitor and period.",
    "SIGINT, SIGTERM and SIGHUP are forwarded to the child process group. "
    "SIGKILL of the monitor cannot be forwarded."};
  for (size_t i = 0; i < limitations.size(); ++i) {
    if (i)
      out << ',';
    quoted(out, limitations[i]);
  }
  out << "],\n\"samples\":[\n";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i)
      out << ",\n";
    sampleJson(out, samples[i]);
  }
  out << "\n]}\n";
  out.close();
  if (!out)
    throw std::runtime_error("output write failed");
  if (rename(temporary.c_str(), output.c_str()) != 0)
    throw std::runtime_error(std::string("output rename failed: ") +
                             strerror(errno));
}

int main(int argc, char **argv) {
  std::string output, fixture;
  int period_ms = 20, command_at = argc;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--") {
        command_at = i + 1;
        break;
      }
      if ((arg == "--output" || arg == "--period-ms" ||
           arg == "--snapshot-proc-dir") &&
          i + 1 < argc) {
        const std::string value(argv[++i]);
        if (arg == "--output")
          output = value;
        else if (arg == "--snapshot-proc-dir")
          fixture = value;
        else {
          U64 period;
          if (!parseUnsigned(value, period) || period < 5 || period > 5000)
            throw std::runtime_error("period must be 5..5000 ms");
          period_ms = static_cast<int>(period);
        }
      } else {
        throw std::runtime_error("usage: memory_monitor --output FILE "
                                 "[--period-ms 20] -- COMMAND [ARG ...]");
      }
    }
    if (output.empty() || (fixture.empty() && command_at == argc) ||
        (!fixture.empty() && command_at != argc))
      throw std::runtime_error(
        "require --output and either a command or --snapshot-proc-dir");
    const std::string temporary = output + ".tmp." + std::to_string(getpid());
    const int output_fd =
      open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (output_fd < 0)
      throw std::runtime_error(std::string("cannot create output: ") +
                               strerror(errno));
    close(output_fd);
    std::vector<std::string> command;
    for (int i = command_at; i < argc; ++i)
      command.emplace_back(argv[i]);
    const auto start = Clock::now();
    const double epoch_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::system_clock::now().time_since_epoch())
        .count();
    std::vector<Sample> samples;
    const auto sample = [&](const std::string &proc_dir, bool pre_exec) {
      const auto begin = Clock::now();
      Sample s = readProc(proc_dir);
      s.begin_ms = milliseconds(start, begin);
      s.end_ms = milliseconds(start, Clock::now());
      s.pre_exec = pre_exec;
      samples.push_back(s);
    };
    if (!fixture.empty()) {
      sample(fixture, false);
      writeReport(output, temporary, samples, command, 0, period_ms, 0, false,
                  milliseconds(start, Clock::now()), epoch_ms, true);
      return 0;
    }
    struct sigaction action = {};
    action.sa_handler = forwardSignal;
    sigemptyset(&action.sa_mask);
    for (int signal : {SIGINT, SIGTERM, SIGHUP})
      if (sigaction(signal, &action, nullptr) != 0)
        throw std::runtime_error("cannot install signal handler");
    if (::signal(SIGPIPE, SIG_IGN) == SIG_ERR)
      throw std::runtime_error("cannot ignore SIGPIPE");
    int gate[2];
    if (pipe2(gate, O_CLOEXEC) != 0)
      throw std::runtime_error("pipe2 failed");
    const pid_t pid = fork();
    if (pid < 0)
      throw std::runtime_error("fork failed");
    if (pid == 0) {
      close(gate[1]);
      setpgid(0, 0);
      for (int signal : {SIGINT, SIGTERM, SIGHUP, SIGPIPE})
        ::signal(signal, SIG_DFL);
      char go = 0;
      ssize_t n;
      do {
        n = read(gate[0], &go, 1);
      } while (n < 0 && errno == EINTR);
      close(gate[0]);
      if (n != 1)
        _exit(125);
      execvp(argv[command_at], argv + command_at);
      std::fprintf(stderr, "memory_monitor: exec failed: %s\n",
                   strerror(errno));
      _exit(127);
    }
    child_pid = static_cast<sig_atomic_t>(pid);
    close(gate[0]);
    setpgid(pid, pid);
    const std::string proc_dir = "/proc/" + std::to_string(pid);
    sample(proc_dir, true);
    if (forwarded_signal)
      forwardSignal(forwarded_signal);
    const char go = 1;
    ssize_t sent;
    do {
      sent = write(gate[1], &go, 1);
    } while (sent < 0 && errno == EINTR);
    close(gate[1]);
    if (sent != 1)
      kill(-pid, SIGTERM);
    int status = 0;
    struct rusage usage = {};
    bool waited = false;
    for (;;) {
      const pid_t done = wait4(pid, &status, WNOHANG, &usage);
      if (done == pid) {
        waited = true;
        break;
      }
      if (done < 0 && errno != EINTR)
        throw std::runtime_error("wait4 failed");
      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
      sample(proc_dir, false);
    }
    child_pid = 0;
    writeReport(output, temporary, samples, command, pid, period_ms, status,
                waited, milliseconds(start, Clock::now()), epoch_ms, false,
                usage.ru_maxrss);
    if (WIFEXITED(status))
      return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
      return 128 + WTERMSIG(status);
    return 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "memory_monitor: %s\n", error.what());
    if (child_pid > 0) {
      const pid_t pid = static_cast<pid_t>(child_pid);
      if (kill(-pid, SIGKILL) != 0)
        kill(pid, SIGKILL);
      int status;
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
    }
    return 1;
  }
}

#else
int main() {
  std::fputs("memory_monitor requires Linux or Android /proc support\n",
             stderr);
  return 77;
}
#endif
