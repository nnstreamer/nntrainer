#!/usr/bin/env python3
"""Build a temporary two-run variant of the existing standalone CausalLM CLI."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--build-dir', type=Path)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--clang-format', type=Path,
                    help='Optional clang-format 14 binary for generated source')
args = parser.parse_args()
root = args.repo.resolve()
build = (args.build_dir or root / 'builddir_app/htp').resolve()
out = (args.output_dir or Path(__file__).resolve().parent).resolve()
out.mkdir(parents=True, exist_ok=True)
original_path = root / 'Applications/CausalLM/main.cpp'
original = original_path.read_text()
begin = original.index('#if defined(_WIN32)\n    model->run(')
end = original.index('\n#endif', begin) + len('\n#endif')
run_call = original[begin:end]

replacement = r'''
    // Test-only wrapper: one factory object, one weight load, two complete runs.
    if (do_sample)
      throw std::runtime_error("repeat check requires do_sample=false");
    if (nntr_cfg.value("num_to_generate", 0u) != 64u)
      throw std::runtime_error("repeat check requires num_to_generate=64");
    if (nntr_cfg.value("skip_prefill", false))
      throw std::runtime_error("repeat check requires skip_prefill=false");
    if (nntr_cfg.value("batch_size", 0u) != 1u ||
        nntr_cfg.value("init_seq_len", 0u) != 2048u ||
        model->prefillPlaneFor(2048u) != 2048u)
      throw std::runtime_error("repeat check requires batch1/2048-row plane");
    const char *prefix_env = std::getenv("NNTR_REPEAT_PREFIX");
    const std::string prefix = prefix_env ? prefix_env : "repeat-prompt";
    std::vector<unsigned int> expected_ids, expected_input;
    json summary;
    summary["skip_prefill"] = false;
    summary["configured_decode_tokens"] = 64;
    summary["expected_dumped_ids"] = 65;
    summary["expected_token_trace_output_ids"] = 64;
    summary["token_trace_skips_prefill_generated_id"] = true;
    summary["runs"] = json::array();
    for (unsigned pass = 1; pass <= 2; ++pass) {
      const std::string stem = prefix + "-run" + std::to_string(pass);
      const std::string ids_path = stem + ".ids";
      const std::string trace_path = stem + ".trace.json";
#ifdef _WIN32
      if (_putenv_s("NNTR_GENERATED_IDS", ids_path.c_str()) ||
          _putenv_s("NNTR_HTP_TOKEN_TRACE", trace_path.c_str()))
#else
      if (setenv("NNTR_GENERATED_IDS", ids_path.c_str(), 1) ||
          setenv("NNTR_HTP_TOKEN_TRACE", trace_path.c_str(), 1))
#endif
        throw std::runtime_error("could not set per-run trace paths");
      std::fprintf(stderr, "[REPEAT-CHECK] begin run=%u model=%p\n", pass,
                   static_cast<const void *>(model.get()));
RUN_CALL
      const auto metrics = model->getPerformanceMetrics();
      if (metrics.prefill_tokens != 2048u || metrics.generation_tokens != 64u ||
          model->prefillPlaneFor(2048u) != 2048u)
        throw std::runtime_error("expected 2048 prefill rows/64 decode steps");
      std::ifstream ids_file(ids_path);
      if (!ids_file)
        throw std::runtime_error("missing generated-ID file: " + ids_path);
      std::vector<unsigned int> ids;
      unsigned int id;
      while (ids_file >> id)
        ids.push_back(id);
      if (!ids_file.eof())
        throw std::runtime_error("invalid generated-ID file: " + ids_path);
      std::ifstream trace_file(trace_path);
      if (!trace_file)
        throw std::runtime_error("missing token trace: " + trace_path);
      json trace;
      trace_file >> trace;
      const auto input_ids =
        trace.at("input_ids").get<std::vector<unsigned int>>();
      const auto trace_ids =
        trace.at("output_ids").get<std::vector<unsigned int>>();
      // CausalLM always dumps prefill's id_list before the decode loop. With
      // skip_prefill=false this is one generated token, followed by64 decode
      // tokens. Its legacy token trace includes prefill's token only when
      // init_len < INIT_SEQ_LEN; our2048-row prompt fills the2048-row plane.
      if (input_ids.size() != 2048 || ids.size() != 65 || trace_ids.size() != 64)
        throw std::runtime_error(
          "expected 2048 input IDs, 65 dumped IDs (prefill1+decode64), "
          "and64 token-trace IDs");
      const std::vector<unsigned int> decode_ids(ids.begin() + 1, ids.end());
      if (trace_ids != decode_ids)
        throw std::runtime_error("token trace differs from dumped decode suffix");
      if (pass == 1) {
        expected_ids = ids;
        expected_input = input_ids;
      } else if (expected_ids != ids || expected_input != input_ids) {
        throw std::runtime_error("repeated prompt input/output IDs differ");
      }
      summary["runs"].push_back({{"run", pass},
                                  {"input_tokens", input_ids.size()},
                                  {"prefill_generated_tokens", 1},
                                  {"decode_tokens", metrics.generation_tokens},
                                  {"dumped_generated_ids", ids.size()},
                                  {"token_trace_output_ids", trace_ids.size()},
                                  {"ids", ids_path},
                                  {"token_trace", trace_path}});
      std::fprintf(stderr,
                   "[REPEAT-CHECK] complete run=%u input=%zu prefill_out=1 "
                   "decode=%u dumped_ids=%zu trace_ids=%zu\n",
                   pass, input_ids.size(), metrics.generation_tokens, ids.size(),
                   trace_ids.size());
    }
    summary["input_ids_equal"] = true;
    summary["output_ids_equal"] = true;
    std::ofstream summary_file(prefix + ".summary.json");
    summary_file << summary.dump(2) << '\n';
    if (!summary_file)
      throw std::runtime_error("could not write repeat summary");
    std::fprintf(stderr,
                 "[REPEAT-CHECK] PASS same model, 2x2048 input, "
                 "each prefill1+decode64=65 dumped IDs, "
                 "identical input/output IDs and matching64-ID trace suffix\n");
'''.replace('RUN_CALL', run_call)
source = original[:begin] + replacement + original[end:]
source = source.replace('#include <string>', '#include <stdexcept>\n#include <string>', 1)
if args.clang_format:
    version = subprocess.check_output([str(args.clang_format), '--version'],
                                      text=True)
    if 'version 14.' not in version:
        parser.error('--clang-format must be version 14')
    source = subprocess.check_output(
        [str(args.clang_format), '--style=file',
         '--assume-filename=' + str(original_path)], input=source, text=True)
cpp = out / 'repeat_prompt_check.cpp'
obj = out / 'repeat_prompt_check.o'
binary = out / 'repeat_prompt_check'
cpp.write_text(source)
(out / 'standalone-test-only.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), source.splitlines(keepends=True),
    fromfile='Applications/CausalLM/main.cpp', tofile=cpp.name)))

records = json.loads((build / 'compile_commands.json').read_text())
record = next(r for r in records if Path(r['file']).resolve() == original_path)
compile_command, skip = [], False
for arg in shlex.split(record['command']):
    if skip:
        skip = False
        continue
    if arg in ('-o', '-MQ', '-MF'):
        skip = True
        continue
    if arg in ('-MD', '-MMD'):
        continue
    compile_command.append(str(cpp) if arg == record['file'] else arg)
compile_command += ['-o', str(obj)]
commands = subprocess.check_output(
    ['ninja', '-t', 'commands', 'nntr_causallm'], cwd=build, text=True)
link_command = None
for line in commands.splitlines():
    parts = shlex.split(line)
    if '-o' in parts and parts[parts.index('-o') + 1] == 'nntr_causallm':
        link_command = parts
assert link_command is not None
link_command[link_command.index('-o') + 1] = str(binary)
objects_replaced = 0
for i, arg in enumerate(link_command):
    if arg.endswith('.o') and 'main.cpp' in arg:
        link_command[i] = str(obj)
        objects_replaced += 1
    if arg.startswith('-Wl,-rpath,'):
        link_command[i] = '-Wl,-rpath,$ORIGIN'
assert objects_replaced == 1
manifest = dict(source_original=str(original_path),
                source_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                compile_command=compile_command, link_command=link_command,
                build_directory=str(build), no_production_source_changes=True,
                expected_semantics=dict(skip_prefill=False, batch_size=1,
                                        init_seq_len=2048, input_ids=2048,
                                        num_to_generate=64, decode_steps=64,
                                        prefill_generated_ids=1,
                                        dumped_generated_ids=65,
                                        token_trace_output_ids=64,
                                        token_trace_mapping='dumped_ids[1:]'))
for name, command, cwd in [('compile', compile_command, record['directory']),
                           ('link', link_command, build)]:
    result = subprocess.run(command, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / (name + '.log')).write_text(result.stdout)
    print(name, result.returncode)
    result.check_returncode()
manifest['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
(out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(binary, manifest['binary_sha256'])
