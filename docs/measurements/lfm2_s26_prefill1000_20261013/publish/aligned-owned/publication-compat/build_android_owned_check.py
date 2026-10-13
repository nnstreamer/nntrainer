from pathlib import Path
import json,hashlib,subprocess,difflib
HERE=Path(__file__).resolve().parent
PUB=Path('/tmp/lfm2-prefill1000-20261013/publish')
SRC=HERE/'android-checker-source';SRC.mkdir(exist_ok=True)
HEADER=PUB/'test/htp/host/expert_host_staging_fixture.h'
SOURCE=PUB/'test/htp/host/owned_envelope_check.cpp'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
header=HEADER.read_text().replace('#include <future>','#include <future>\n#include <string>',1)
needle='''static inline int readAll(int fd, void *dst, size_t n, uint64_t off) {'''
helper='''// Private standalone adaptation: choose a writable Android test directory.
// File contents, production headers and all assertions remain unchanged.
static inline int ownedTempFile(const char *name) {
  const char *directory = std::getenv("NNTR_TEST_TMPDIR");
  if (!directory || !*directory) {
#if defined(__ANDROID__)
    directory = "/data/local/tmp";
#else
    directory = "/tmp";
#endif
  }
  std::string pattern = std::string(directory) + "/" + name;
  std::vector<char> path(pattern.begin(), pattern.end());
  path.push_back(0);
  int fd = mkstemp(path.data());
  if (fd >= 0)
    unlink(path.data());
  return fd;
}
'''
assert needle in header;header=header.replace(needle,helper+needle,1)
old='''    char path[] = "/tmp/nntr-host-stage-XXXXXX";
    fd = mkstemp(path);
    require(fd >= 0);
    unlink(path);'''
new='''    fd = ownedTempFile("nntr-host-stage-XXXXXX");
    require(fd >= 0);'''
assert old in header;header=header.replace(old,new,1)
source=SOURCE.read_text()
old='''  char path[] = "/tmp/nntr-owned-direct-XXXXXX";
  int fd = mkstemp(path);
  require(fd >= 0);
  unlink(path);'''
new='''  int fd = ownedTempFile("nntr-owned-direct-XXXXXX");
  require(fd >= 0);'''
assert old in source;source=source.replace(old,new,1)
old='''  char next_path[] = "/tmp/nntr-owned-reused-XXXXXX";
  fd = mkstemp(next_path);
  require(fd == reused);
  unlink(next_path);'''
new='''  fd = ownedTempFile("nntr-owned-reused-XXXXXX");
  require(fd == reused);'''
assert old in source;source=source.replace(old,new,1)
(SRC/HEADER.name).write_text(header);(SRC/SOURCE.name).write_text(source)
formatter='/tmp/lfm2-prefill1000-20261013/format/clang_format/data/bin/clang-format'
subprocess.run([formatter,'-i',str(SRC/HEADER.name),str(SRC/SOURCE.name)],check=True)
patch=''
for p in [HEADER,SOURCE]:
 patch+=''.join(difflib.unified_diff(p.read_text().splitlines(True),(SRC/p.name).read_text().splitlines(True),fromfile='pub/'+p.name,tofile='android-checker/'+p.name))
(HERE/'android-checker-tempdir-only.patch').write_text(patch)
flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(SRC),'-I'+str(PUB/'nntrainer/utils'),'-I'+str(PUB/'nntrainer/tensor/htp_backend'),str(SRC/SOURCE.name),'-Wl,--wrap=pread','-Wl,--wrap=free','-Wl,--wrap=posix_memalign']
android=Path('/home/jungone-lee/workspace/nntrainer/android-ndk-r30/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++')
binary=HERE/'owned_envelope_check-android'
command=[str(android)]+flags+['-static-libstdc++','-o',str(binary)]
with (HERE/'android-checker-build.log').open('w') as log:
 subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
# Execute exactly the adapted fixture on the host; Android execution is parent-owned.
host=HERE/'owned_envelope_check-host-adapted'
with (HERE/'android-checker-host-validation.log').open('w') as log:
 subprocess.run(['c++']+flags+['-o',str(host)],check=True,stdout=log,stderr=subprocess.STDOUT)
 subprocess.run([str(host)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=60)
report={'binary':str(binary),'sha256':sha(binary),'bytes':binary.stat().st_size,'command':command,'source_sha256':{p.name:sha(p) for p in SRC.iterdir() if p.is_file()},'production_header_sha256':{str(p):sha(p) for p in [PUB/'nntrainer/utils/weight_direct_file.h',PUB/'nntrainer/tensor/htp_backend/htp_expert_host_staging.h',PUB/'nntrainer/tensor/htp_backend/htp_qs2_reader.h']},'tempdir_only_patch':str(HERE/'android-checker-tempdir-only.patch'),'host_adapted_run':'PASS','parent_device_command':'NNTR_TEST_TMPDIR=<writable device directory> ./owned_envelope_check-android','expected':'PASS owned envelopes:33662 assertions','device_run_by_agent':False}
(HERE/'android-checker-artifacts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['binary','sha256','bytes','host_adapted_run']},indent=2))
