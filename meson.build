project('nntrainer', 'c', 'cpp',
  version: '0.6.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

# Set version info
nntrainer_version = meson.project_version()
nntrainer_version_split = nntrainer_version.split('.')

# CMake module
cmake = import('cmake')
fs = import('fs')

extra_defines = ['-DMIN_CPP_VERSION=201703L']

add_project_arguments('-DVERSION="' + nntrainer_version + '"', language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MAJOR=' + nntrainer_version_split[0], language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MINOR=' + nntrainer_version_split[1], language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MICRO=' + nntrainer_version_split[2], language: ['c', 'cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
cxx_compiler_id = cxx.get_id()

powershell_prog = find_program('powershell', required: (build_machine.system() == 'windows'))

# Obtains number of cores
nproc_prog = find_program('nproc',
                          required: (get_option('platform') == 'android') and (build_machine.system() != 'windows'))

prog_win_cmd = find_program('cmd.exe', required: build_machine.system() == 'windows')

if get_option('platform') == 'tizen'
  # Pass __TIZEN__ to the compiler
  add_project_arguments('-D__TIZEN__=1', language:['c','cpp'])
  add_project_arguments('-DTIZENVERSION=@0@'.format(get_option('tizen-version-major')), language: ['c', 'cpp'])
  add_project_arguments('-DTIZENVERSIONMINOR=@0@'.format(get_option('tizen-version-minor')), language: ['c', 'cpp'])

  if get_option('enable-tizen-feature-check')
    add_project_arguments('-D__FEATURE_CHECK_SUPPORT__', language: ['c', 'cpp'])
  endif
endif

if get_option('enable_encoder')
  add_project_arguments('-DENABLE_ENCODER=1', language: ['c', 'cpp'])
endif

warning_flags = [
  '-Wredundant-decls',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Winit-self',
  '-Waddress',
  '-Wvla',
  '-Wpointer-arith',
  '-Wno-error=varargs',
  '-Wdefaulted-function-deleted',
  '-ftree-vectorize',
  '-Wno-unused-variable',
  '-Wno-comment'
]

if cxx_compiler_id == 'clang'
  warning_flags += '-Wno-uninitialized'
  warning_flags += '-Wno-format-nonliteral'
  warning_flags += '-Wno-varargs'
  warning_flags += '-Wno-missing-braces'
else
  warning_flags += '-Wno-maybe-uninitialized'
endif

warning_c_flags = [
  '-Wmissing-declarations',
  '-Wmissing-include-dirs',
  '-Wmissing-prototypes',
  '-Wnested-externs',
  '-Waggregate-return',
  '-Wold-style-definition',
  '-Wdeclaration-after-statement',
  '-Wno-error=varargs'
]

arch = host_machine.cpu_family()

target = target_machine.cpu_family()


if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  message('Build for ARM architecture')
  if arch == 'arm'
    extra_defines += '-DARMV7=1'
  endif
elif arch == 'x86' or arch == 'x86_64'
  message('Build for X86 architecture')
  if cxx_compiler_id == 'msvc'
    add_project_arguments(['/arch:AVX2'], language: ['c','cpp'])
  else
    add_project_arguments(['-march=native'], language: ['c','cpp'])
    add_project_arguments(['-mavx2', '-mfma'], language: ['c','cpp'])
    message('-march=native added for AVX hardware acceleration.')
  endif
endif

if get_option('enable-fp16')
   if get_option('platform') == 'android'
     add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
     extra_defines += '-DENABLE_FP16=1'
     extra_defines += '-DUSE__FP16=1'
   elif arch == 'aarch64'
     ## About FP16 in GCC (from GCC-9.1 manual)
     # https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Half-Precision.html
     # On ARM and AArch64 targets, GCC supports half-precision (16-bit) floating point
     #   via the __fp16 type defined in the ARM C Language Extensions.
     # On ARM systems, you must enable this type explicitly with the -mfp16-format
     #   command-line option in order to use it.
     ## About FP16-SIMD in aarch64
     # FP16-SIMD is supported since armv8.2. If you enable this forcibly, it won't be
     # comaptible with armv8.0 machines.
     if cxx.has_argument('-mfp16-format=ieee')
       add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
       add_project_arguments('-march=armv8.2-a+fp16', language: ['c', 'cpp'])
     else
       message ('The compiler does not support -mfp16-format=ieee. However, according to https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Half-Precision.html, gcc may use IEEE fp16 anyway. Thus, we will proceed without the option for FP16 support.')
     endif
     extra_defines += '-DENABLE_FP16=1'
     extra_defines += '-DUSE__FP16=1'
   elif arch == 'arm'
     ## About FP16-SIMD in arm
     # FP16-SIMD is supported since armv8.2.
     # Thus, even if fp16 is force-enabled, NEON is off.
     if cxx.has_argument('-mfp16-format=ieee')
       add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
       extra_defines += '-DENABLE_FP16=1'
       extra_defines += '-DUSE__FP16=1'
     else
       error ('The compiler does not support -mfp16-format=ieee')
     endif
   elif arch == 'x86_64'
     if cc.version().version_compare('>=12.1.0')
       message ('Float16 for x86_64 enabled. Modern gcc-x64 generally supports float16 with _Float16.')
       extra_defines += '-DENABLE_FP16=1'
     else
       warning ('Float16 for x86_64 enabled. However, software emulation is applied for fp16, making it slower and inconsistent. Use GCC 12+ for FP16 support. This build will probably fail unless you bring a compiler that supports fp16 for x64.')
     endif
   elif arch == 'riscv64'
     error ('RISCV64 RVV support and fp16 support is not yet implemented.')
   else
     error ('FP16 support for this arch is not yet implemented.')
   endif  
endif

if get_option('enable-mmap')
  message ('MMAP enabled')
  extra_defines += '-DUSE_MMAP=1'
endif

dummy_dep = dependency('', required: false)
third_party_build_dir = 'build_' + cxx_compiler_id
# CLBlast
clblast_dep = dummy_dep

if get_option('enable-opencl')
  message ('OpenCL build is now enabled and will function only if an OpenCL-supported GPU is available. Using CLBlast as the OpenCL backend.')
  extra_defines += '-DENABLE_OPENCL=1'

  if get_option('platform') == 'android'
    message('preparing opencl')
    run_command([meson.current_source_dir() / 'jni' / 'prepare_opencl.sh', meson.current_build_dir()], check: true)
    clblast_root = meson.current_source_dir() / 'third_party' / 'CLBlast'
    opencl_root = meson.current_build_dir() / 'opencl'
    clblast_dep = declare_dependency()
  else
    message('Configuring')
    clblast_dir_relative = 'third_party' / 'CLBlast'
    clblast_include_dir_relative = clblast_dir_relative / 'include'
    clblast_dir_absolute = meson.current_source_dir() / clblast_dir_relative
    clblast_build_dir_relative = clblast_dir_relative / third_party_build_dir
    clblast_build_dir_absolute = meson.current_source_dir() / clblast_build_dir_relative

    if fs.exists (clblast_build_dir_absolute)
      message('Skip compiling CLBlast (using previous build)')
    else
      message('Setup CLBlast')
      clblast_setup_command = ['cmake', '-B', clblast_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=17', '-DCMAKE_POLICY_VERSION_MINIMUM=3.10']

      if cxx_compiler_id == 'clang'
        clblast_setup_command += '-G Ninja'
        clblast_setup_command += '-DCMAKE_C_COMPILER=clang'
        clblast_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
      endif

      clblast_setup_command += clblast_dir_absolute

      message(clblast_setup_command)
      run_command(clblast_setup_command, check: true)

      message('Compile CLBlast')
      clblast_compile_command = ['cmake' , '--build', clblast_build_dir_absolute, '--config', 'Release', '-j']
      message(clblast_compile_command)
      run_command(clblast_compile_command, check: true)
    endif

    if host_machine.system() == 'windows'
      if cxx_compiler_id == 'msvc'
        clblast_lib_dir_absolute = clblast_build_dir_absolute / 'Release'
        clblast_lib = clblast_lib_dir_absolute / 'clblast.lib'
      else
        clblast_lib_dir_absolute = clblast_build_dir_absolute
        clblast_lib = clblast_lib_dir_absolute / 'clblast.lib'
      endif
    else
      clblast_lib_dir_absolute = clblast_build_dir_absolute
      clblast_lib = clblast_lib_dir_absolute / 'libclblast.so'
    endif

    clblast_include = include_directories(clblast_include_dir_relative)
    clblast_dep = declare_dependency(include_directories: clblast_include, link_args:[clblast_lib])
  endif
endif


if get_option('opencl-kernel-path') != ''
  message ('OpenCL kernel path set to: @0@'.format(get_option('opencl-kernel-path')))
  extra_defines += '-DOPENCL_KERNEL_PATH=@0@'.format(get_option('opencl-kernel-path'))
endif

if get_option('enable-biqgemm')
  # In order to enable biqgemm, BiQGEMM lib, which is header-only library, is required.
  # This meson tries to find its installation.
  #   1. Checking 'prefix'/'includedir'/BiQGEMM
  #   2. Checking path meson-option specifies
  biqgemm_path = join_paths(get_option('prefix'), get_option('includedir'), 'BiQGEMM')
  if cxx.has_header('BiQGEMM.h', args: '-I'+biqgemm_path)
      message('[lib:biqgemm] biqgemm header is found successfully')
      extra_defines += '-DENABLE_BIQGEMM=1'
      biqgemm_inc = include_directories(biqgemm_path)
  else
      # relative path from biqgemm is assumed
      biqgemm_path = get_option('biqgemm-path')
      message('[lib:biqgemm] fallback: finding biqgemm from user-path :' + biqgemm_path)
      if fs.is_dir(biqgemm_path) 
        message('[lib:biqgemm] biqgemm header is found successfully')
        extra_defines += '-DENABLE_BIQGEMM=1'
        biqgemm_inc = include_directories(biqgemm_path)
      else
        error ('BiQGEMM cannot be enabled without BiQGEMM library.')
      endif
  endif
endif # end of enable-biqgemm

if cc.get_id() != 'msvc' and cxx.get_id() != 'msvc'
  foreach extra_arg : warning_flags
    if cc.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'c')
    endif
    if cxx.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'cpp')
    endif
  endforeach

  foreach extra_arg : warning_c_flags
    if cc.has_argument (extra_arg)
      add_project_arguments([extra_arg], language: 'c')
    endif
  endforeach
endif

# Set install path
nntrainer_prefix = get_option('prefix')
if get_option('platform') != 'android'
  nntrainer_libdir = nntrainer_prefix / get_option('libdir')
  nntrainer_bindir = nntrainer_prefix / get_option('bindir')
  nntrainer_includedir = nntrainer_prefix / get_option('includedir') / 'nntrainer'
  nntrainer_confdir = get_option('sysconfdir')
  application_install_dir = nntrainer_bindir / 'applications'
  nntrainer_fsudir = '/tmp'
else
  nntrainer_prefix = meson.current_build_dir() / 'android_build_result'
  # @todo arch has to be option
  nntrainer_libdir = nntrainer_prefix / 'lib'
  nntrainer_includedir = nntrainer_prefix / 'include' / 'nntrainer'
  nntrainer_bindir = nntrainer_prefix / 'bin'
  nntrainer_confdir = nntrainer_prefix / 'conf'
  application_install_dir = nntrainer_prefix / 'examples'
  nntrainer_fsudir = '/data/local/tmp'
endif

# handle fsu options
if get_option('enable-fsu')
  nntrainer_enable_fsu = 'true'
else
  nntrainer_enable_fsu = 'false'
endif

if get_option('fsu-path') != ''
  nntrainer_fsudir = get_option('fsu-path')
endif

# handle resources
nntrainer_resdir = meson.current_build_dir() / 'res'

if build_machine.system() == 'windows'
  nntrainer_resdir_win = nntrainer_resdir.replace('/', '\\')
  if not fs.exists (nntrainer_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntrainer_resdir_win], check: true)
  endif
else
  run_command(['mkdir', '-p', nntrainer_resdir], check: true)
endif

if get_option('install-app')
# add a script to install resources from installs to application_install_dir
meson.add_install_script(
  'sh', '-c', 'cp -r @0@ ${DESTDIR}@1@'.format(nntrainer_resdir, application_install_dir)
)
endif

# Set default configuration
nntrainer_conf = configuration_data()
nntrainer_conf.set('VERSION', meson.project_version())
nntrainer_conf.set('PREFIX', nntrainer_prefix)
nntrainer_conf.set('EXEC_PREFIX', nntrainer_bindir)
nntrainer_conf.set('LIB_INSTALL_DIR', nntrainer_libdir)
nntrainer_conf.set('PLUGIN_INSTALL_PREFIX', nntrainer_libdir / 'nntrainer')
nntrainer_conf.set('INCLUDE_INSTALL_DIR', nntrainer_includedir / '..')
nntrainer_conf.set('FLASH_STORAGE_UTILIZATION', nntrainer_enable_fsu)
nntrainer_conf.set('FLASH_STORAGE_UTILIZATION_PATH', nntrainer_fsudir)

ml_api_common_flag = '-DML_API_COMMON=0'

# if ml-api-support is disabled, enable dummy common api interfaces and disable related dependencies.
ml_api_common_dep = dependency(get_option('capi-ml-common-actual'), required : get_option('ml-api-support').enabled())
nnstreamer_capi_dep = dummy_dep
if (ml_api_common_dep.found())
  nnstreamer_capi_dep = dependency(get_option('capi-ml-inference-actual'), required : get_option('ml-api-support').enabled())
  if (nnstreamer_capi_dep.found())
    nntrainer_conf.set('CAPI_ML_COMMON_DEP', get_option('capi-ml-common-actual'))
    extra_defines += '-DML_API_COMMON=1'
    ml_api_common_flag = '-DML_API_COMMON=1'
    extra_defines += '-DNNSTREAMER_AVAILABLE=1'
    # accessing this variable when dep_.not_found() remains hard error on purpose
    supported_nnstreamer_capi = nnstreamer_capi_dep.version().version_compare('>=1.7.0')
    if not supported_nnstreamer_capi
      extra_defines += '-DUNSUPPORTED_NNSTREAMER=1'
      warning('capi-nnstreamer version is too old, we do not know if it works with older nnstreamer version')
    endif
  else
    # if nnstreamer_capi is not there and ml-api-support is "auto", disable it.
    message ('ml-api-support is disabled although capi-ml-api-common is found: capi-ml-api-inference is not available and ml-api-support is configured to be auto')
    nntrainer_conf.set('CAPI_ML_COMMON_DEP', '')
    extra_defines += '-DML_API_COMMON=0'
  endif
else
  nntrainer_conf.set('CAPI_ML_COMMON_DEP', '')
  extra_defines += '-DML_API_COMMON=0'
endif

# OpenBLAS
blas_dep = dummy_dep
# Dependencies
if get_option('enable-cublas')
  extra_defines += '-DUSE_CUBLAS=1'
endif

if get_option('enable-blas')
  extra_defines += '-DUSE_BLAS=1'

  if get_option('platform') == 'android'
    message('preparing blas')
    run_command([meson.current_source_dir() / 'jni' / 'prepare_openblas.sh', meson.current_build_dir()], check: true)
    blas_root = meson.current_build_dir() / 'openblas'
    blas_dep = declare_dependency(include_directories: [ 'openblas/include' ])
  else
    blas_dep = dependency('openblas', required : false)

    if not blas_dep.found()
      message('Cannot find OpenBLAS dependency in system')
      openblas_dir_relative = 'third_party' / 'OpenBLAS'
      openblas_dir_absolute = meson.current_source_dir() / openblas_dir_relative
      openblas_build_dir_relative = openblas_dir_relative / third_party_build_dir
      openblas_build_dir_absolute = meson.current_source_dir() / openblas_build_dir_relative

      if fs.exists (openblas_build_dir_absolute)
        message('Skip compiling OpenBLAS (using previous build)')
      else
        message('Setup OpenBLAS')
        openblas_setup_command = ['cmake', '-B', openblas_build_dir_absolute, '-DNUM_THREADS=' + get_option('openblas-num-threads').to_string(), '-DBUILD_TESTING=False', '-DBUILD_BENCHMARKS=False', '-DBUILD_WITHOUT_LAPACK=True', '-DNOFORTRAN=True', '-DCMAKE_BUILD_TYPE=Release']

        if cxx_compiler_id == 'clang'
          openblas_setup_command += '-G Ninja'
          openblas_setup_command += '-DCMAKE_C_COMPILER=clang'
          openblas_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
        endif

        openblas_setup_command += openblas_dir_absolute

        message(openblas_setup_command)
        run_command(openblas_setup_command, check: true)

        message('Compile OpenBLAS')
        openblas_compile_command = ['cmake' , '--build', openblas_build_dir_absolute, '--config', 'Release', '-j']
        message(openblas_compile_command)
        run_command(openblas_compile_command, check: true)
      endif

      if host_machine.system() == 'windows'
        if cxx_compiler_id == 'msvc'
          openblas_lib = openblas_build_dir_absolute / 'lib' / 'RELEASE' / 'openblas.lib'
        else
          openblas_lib = openblas_build_dir_absolute / 'lib' / 'openblas.lib'
        endif
      else
        openblas_lib = openblas_build_dir_absolute / 'lib' / 'libopenblas.so'
      endif

      openblas_include = include_directories(openblas_dir_relative, openblas_build_dir_relative)
      blas_dep = declare_dependency(include_directories: openblas_include, link_args:[openblas_lib])
    else
      message('OpenBLAS dependency found in system')
    endif
  endif

  if blas_dep.found()
    if get_option('openblas-num-threads') > 0
      extra_defines += '-DBLAS_NUM_THREADS=@0@'.format(get_option('openblas-num-threads'))
      message('set openblas num threads=@0@'.format(get_option('openblas-num-threads')))
    endif
  endif
  extra_defines += '-DHGEMM_EXPERIMENTAL_KERNEL=@0@'.format(get_option('hgemm-experimental-kernel'))
endif

extra_defines += '-DNNTR_NUM_THREADS=@0@'.format(get_option('nntr-num-threads'))
message('set nntrainer num threads=@0@'.format(get_option('nntr-num-threads')))

openmp_dep = dummy_dep
if get_option('enable-openmp')
  openmp_dep = dependency('openmp')
  if get_option('omp-num-threads') > 0
    extra_defines += '-DOMP_NUM_THREADS=@0@'.format(get_option('omp-num-threads'))
    message('set nntrainer omp threads=@0@'.format(get_option('omp-num-threads')))
  endif
endif

if get_option('enable-profile')
  extra_defines += '-DPROFILE=1'
endif

if get_option('enable-trace')
  extra_defines += '-DTRACE=1'
endif

if get_option('enable-debug')
  extra_defines += '-DDEBUG=1'
endif

if get_option('use_gym')
  extra_defines += '-DUSE_GYM=1'
endif

if get_option('enable-logging')
  extra_defines += '-D__LOGGING__=1'
endif

if get_option('enable-benchmarks')
  benchmark_dep = dependency('benchmark', static : true, main : false, required : false)
  if not benchmark_dep.found()
    message('Cannot find benchmark dependency in system')
    benchmark_dir_relative = 'third_party' / 'benchmark'
    benchmark_include_dir_relative = benchmark_dir_relative / 'include'
    benchmark_dir_absolute = meson.current_source_dir() / benchmark_dir_relative
    benchmark_build_dir_relative = benchmark_dir_relative / third_party_build_dir
    benchmark_build_dir_absolute = meson.current_source_dir() / benchmark_build_dir_relative

    if fs.exists (benchmark_build_dir_absolute)
      message('Skip compiling benchmark (using previous build)')
    else
      message('Setup benchmark')
      benchmark_setup_command = ['cmake', '-B', benchmark_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release', '-DBENCHMARK_ENABLE_TESTING=False']

      if cxx_compiler_id == 'clang'
        benchmark_setup_command += '-G Ninja'
        benchmark_setup_command += '-DCMAKE_C_COMPILER=clang'
        benchmark_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
      endif

      benchmark_setup_command += benchmark_dir_absolute

      message(benchmark_setup_command)
      run_command(benchmark_setup_command, check: true)

      message('Compile benchmark')
      benchmark_compile_command = ['cmake' , '--build', benchmark_build_dir_absolute, '--config', 'Release', '-j']
      message(benchmark_compile_command)
      run_command(benchmark_compile_command, check: true)
    endif

    if host_machine.system() == 'windows'
      if cxx_compiler_id == 'msvc'
        benchmark_lib = benchmark_build_dir_absolute / 'src' / 'Release' / 'benchmark.lib'
      else
        benchmark_lib = benchmark_build_dir_absolute / 'src' / 'benchmark.lib'
      endif
    else
      benchmark_lib = benchmark_build_dir_absolute / 'src' / 'libbenchmark.so'
    endif

    benchmark_include = include_directories(benchmark_include_dir_relative)
    benchmark_dep = declare_dependency(include_directories: benchmark_include, link_args:[benchmark_lib])
  endif
endif

# googletest
use_custom_googletest = false
gmock_dep = dependency('gmock', static: true, main: false, required: false)
gtest_dep = dependency('gtest', static: true, main: false, required: false)
gtest_main_dep = dependency('gtest', static: true, main: true, required: false)

if not gmock_dep.found() or not gtest_dep.found() or not gtest_main_dep.found()
  message('Cannot find googletest dependency in system')
  use_custom_googletest = true
  googletest_dir_relative = 'third_party' / 'googletest'
  googletest_dir_absolute = meson.current_source_dir() / googletest_dir_relative
  googletest_build_dir_relative = googletest_dir_relative / third_party_build_dir
  googletest_build_dir_absolute = meson.current_source_dir() / googletest_build_dir_relative

  if fs.exists (googletest_build_dir_absolute)
    message('Skip compiling googletest (using previous build)')
  else
    message('Setup googletest')
    googletest_setup_command = ['cmake', '-B', googletest_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release']

    if cxx_compiler_id == 'clang'
      googletest_setup_command += '-G Ninja'
      googletest_setup_command += '-DCMAKE_C_COMPILER=clang'
      googletest_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
    endif

    googletest_setup_command += googletest_dir_absolute

    message(googletest_setup_command)
    run_command(googletest_setup_command, check: true)

    message('Compile googletest')
    googletest_compile_command = ['cmake' , '--build', googletest_build_dir_absolute, '--config', 'Release', '-j']
    message(googletest_compile_command)
    run_command(googletest_compile_command, check: true)
  endif

  if host_machine.system() == 'windows'
    if cxx_compiler_id == 'msvc'
      googletest_lib_dir_absolute = googletest_build_dir_absolute / 'lib' / 'Release'
      gtest_lib = googletest_lib_dir_absolute / 'gtest.lib'
      gtest_main_lib = googletest_lib_dir_absolute / 'gtest_main.lib'
      gmock_lib = googletest_lib_dir_absolute / 'gmock.lib'
    else
      googletest_lib_dir_absolute = googletest_build_dir_absolute / 'lib'
      gtest_lib = googletest_build_dir_absolute / 'gtest.lib'
      gtest_main_lib = googletest_build_dir_absolute / 'gtest_main.lib'
      gmock_lib = googletest_build_dir_absolute / 'gmock.lib'
    endif
  else
      googletest_lib_dir_absolute = googletest_build_dir_absolute / 'lib'
      gtest_lib = googletest_lib_dir_absolute / 'libgtest.so'
      gtest_main_lib = googletest_lib_dir_absolute / 'libgtest_main.so'
      gmock_lib = googletest_lib_dir_absolute / 'libgmock.so'
  endif

  gtest_include = include_directories(googletest_dir_relative / 'googletest' / 'include')
  gmock_include = include_directories(googletest_dir_relative / 'googlemock' / 'include')

  gmock_dep = declare_dependency(include_directories: gmock_include, link_args:[gmock_lib])
  gtest_dep = declare_dependency(include_directories: gtest_include, link_args:[gtest_lib])
  gtest_main_dep = declare_dependency(include_directories: gtest_include, link_args:[gtest_main_lib, gtest_lib])
endif

if get_option('enable-test') # and get_option('platform') != 'android'
  extra_defines += '-DENABLE_TEST=1'
  if gtest_dep.version().version_compare('<1.10.0')
     extra_defines += '-DGTEST_BACKPORT=1'
  endif
  test_timeout = get_option('test-timeout')
endif

if get_option('reduce-tolerance')
  extra_defines += '-DREDUCE_TOLERANCE=1'
endif

libm_dep = cxx.find_library('m', required: false) # cmath library
libdl_dep = cxx.find_library('dl', required: false) # DL library

thread_dep = dependency('threads') # pthread for tensorflow-lite

# iniparser
use_custom_iniparser = false

if get_option('platform') == 'android'
  iniparser_root = meson.current_source_dir() / 'third_party' / 'iniparser'
  iniparser_dep = declare_dependency()
else
  iniparser_dep = dependency('iniparser', required : false, version : '>=3.2') # iniparser

  if not iniparser_dep.found()
    message('Cannot find iniparser dependency in system')
    use_custom_iniparser = true
    iniparser_dir_relative = 'third_party' / 'iniparser'
    iniparser_include_dir_relative = iniparser_dir_relative / 'src'
    iniparser_dir_absolute = meson.current_source_dir() / iniparser_dir_relative
    iniparser_build_dir_relative = iniparser_dir_relative / third_party_build_dir
    iniparser_build_dir_absolute = meson.current_source_dir() / iniparser_build_dir_relative

    if fs.exists (iniparser_build_dir_absolute)
      message('Skip compiling iniparser (using previous build)')
    else
      message('Setup iniparser')
      iniparser_setup_command = ['cmake', '-B', iniparser_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_DOCS=False', '-DBUILD_EXAMPLES=False', '-DBUILD_TESTING=False']

      if cxx_compiler_id == 'clang'
        iniparser_setup_command += '-G Ninja'
        iniparser_setup_command += '-DCMAKE_C_COMPILER=clang'
        iniparser_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
      endif

      iniparser_setup_command += iniparser_dir_absolute

      message(iniparser_setup_command)
      run_command(iniparser_setup_command, check: true)

      message('Compile iniparser')
      iniparser_compile_command = ['cmake' , '--build', iniparser_build_dir_absolute, '--config', 'Release', '-j']
      message(iniparser_compile_command)
      run_command(iniparser_compile_command, check: true)
    endif

    if host_machine.system() == 'windows'
      if cxx_compiler_id == 'msvc'
        iniparser_lib_dir_absolute = iniparser_build_dir_absolute / 'Release'
        iniparser_lib = iniparser_lib_dir_absolute / 'iniparser.lib'
      else
        iniparser_lib_dir_absolute = iniparser_build_dir_absolute
        iniparser_lib = iniparser_lib_dir_absolute / 'iniparser.lib'
      endif
    else
      iniparser_lib_dir_absolute = iniparser_build_dir_absolute
      iniparser_lib = iniparser_lib_dir_absolute / 'libiniparser.so'
    endif

    iniparser_include = include_directories(iniparser_include_dir_relative)
    iniparser_dep = declare_dependency(include_directories: iniparser_include, link_args:[iniparser_lib])
  endif
endif

# ggml
ggml_dep = dummy_dep

if get_option('enable-ggml')
  message('Configuring ggml')
  extra_defines += '-DENABLE_GGML=1'
  ggml_root = meson.current_source_dir() / 'third_party' / 'ggml'

  if get_option('platform') == 'android'
    message('preparing ggml for Android')
    run_command([meson.current_source_dir() / 'jni' / 'prepare_ggml.sh', ggml_root ], check: true)
  endif

  if get_option('platform') != 'tizen' and get_option('platform') != 'android'
    ggml_dir_relative = 'third_party' / 'ggml'
    ggml_patch_relative = 'third_party' / 'packagefiles' / 'ggml' / '0001-nntrainer-ggml-patch.patch'
    ggml_dir_absolute = meson.current_source_dir() / ggml_dir_relative
    ggml_build_dir_relative = ggml_dir_relative / third_party_build_dir
    ggml_build_dir_absolute = meson.current_source_dir() / ggml_build_dir_relative

    if fs.exists (ggml_build_dir_absolute)
      message('Skip compiling ggml (using previous build)')
    else
      message('Patch ggml')
      ggml_patch_command = ['git', 'apply', '--directory=' + ggml_dir_relative, ggml_patch_relative]
      message(ggml_patch_command)
      run_command(ggml_patch_command, check: true)

      message('Setup ggml')
      ggml_setup_command = ['cmake', '-B', ggml_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_BUILD_TESTS=False', '-DGGML_BUILD_EXAMPLES=False']

      if get_option('platform') == 'android'
        ggml_setup_command += '-DCMAKE_INSTALL_LIBDIR=' + nntrainer_libdir / 'arm64-v8a'
      endif

      if host_machine.system() == 'windows'
        ggml_setup_command += '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True'
      endif

      if cxx_compiler_id == 'clang'
        ggml_setup_command += '-G Ninja'
        ggml_setup_command += '-DCMAKE_C_COMPILER=clang'
        ggml_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
      endif

      ggml_setup_command += ggml_dir_absolute

      message(ggml_setup_command)
      run_command(ggml_setup_command, check: true)

      message('Compile ggml')
      ggml_compile_command = ['cmake' , '--build', ggml_build_dir_absolute, '--config', 'Release', '-j']
      message(ggml_compile_command)
      run_command(ggml_compile_command, check: true)
    endif

    if host_machine.system() == 'windows'
      if cxx_compiler_id == 'msvc'
        ggml_lib_dir_absolute = ggml_build_dir_absolute / 'src' / 'Release'
        ggml_dll_dir_absolute = ggml_build_dir_absolute / 'bin' / 'Release'
        ggml_lib = ggml_lib_dir_absolute / 'ggml.lib'
        ggml_base_lib = ggml_lib_dir_absolute / 'ggml-base.lib'
        ggml_cpu_lib = ggml_lib_dir_absolute / 'ggml-cpu.lib'
      else
        ggml_lib_dir_absolute = ggml_build_dir_absolute / 'src'
        ggml_lib = ggml_lib_dir_absolute / 'ggml.lib'
        ggml_base_lib = ggml_lib_dir_absolute / 'ggml-base.lib'
        ggml_cpu_lib = ggml_lib_dir_absolute / 'ggml-cpu.lib'
      endif
    else
        ggml_lib_dir_absolute = ggml_build_dir_absolute / 'src'
        ggml_lib = ggml_lib_dir_absolute / 'libggml.so'
        ggml_base_lib = ggml_lib_dir_absolute / 'libggml-base.so'
        ggml_cpu_lib = ggml_lib_dir_absolute / 'libggml-cpu.so'
    endif

    ggml_include = include_directories(ggml_dir_relative / 'include', ggml_dir_relative / 'src', ggml_dir_relative / 'src' / 'ggml-cpu')
    ggml_dep = declare_dependency(include_directories: ggml_include, link_args:[ggml_lib, ggml_base_lib, ggml_cpu_lib])
  endif
endif

# Configure the Ruy project (CMake)
if get_option('platform') == 'android'
  ruy_root = meson.current_source_dir() / 'third_party' / 'ruy'
  ruy_dep = declare_dependency()
else
  message('Configuring ruy')
  ruy_dir_relative = 'third_party' / 'ruy'
  ruy_dir_absolute = meson.current_source_dir() / ruy_dir_relative
  ruy_build_dir_relative = ruy_dir_relative / third_party_build_dir
  ruy_build_dir_absolute = meson.current_source_dir() / ruy_build_dir_relative

  if fs.exists (ruy_build_dir_absolute)
    message('Skip compiling ruy (using previous build)')
  else
    message('Setup ruy')
    ruy_setup_command = ['cmake', '-B', ruy_build_dir_absolute, '-DCMAKE_BUILD_TYPE=Release', '-DRUY_MINIMAL_BUILD=True', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']

    if cxx_compiler_id == 'clang'
      ruy_setup_command += '-G Ninja'
      ruy_setup_command += '-DCMAKE_C_COMPILER=clang'
      ruy_setup_command += '-DCMAKE_CXX_COMPILER=clang++'
    endif

    ruy_setup_command += ruy_dir_absolute

    message(ruy_setup_command)
    run_command(ruy_setup_command, check: true)

    message('Compile ruy')
    ruy_compile_command = ['cmake' , '--build', ruy_build_dir_absolute, '--config', 'Release', '-j']
    message(ruy_compile_command)
    run_command(ruy_compile_command, check: true)
  endif

  ruy_include = include_directories(ruy_dir_relative)
  ruy_dep = declare_dependency(include_directories: ruy_include)
endif

if get_option('platform') == 'android'
  message('preparing ml api')
  run_command([meson.current_source_dir() / 'jni' / 'prepare_ml-api.sh', meson.current_build_dir() / 'ml-api-inference'], check: true)
  ml_api_common_root = meson.current_build_dir() / 'ml-api-inference'
  ml_api_inc = ml_api_common_root / 'include'
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'ml-api-common.h', nntrainer_includedir)
  )
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'tizen_error.h', nntrainer_includedir)
  )
  ml_api_common_dep = declare_dependency(include_directories: ['ml-api-inference/include'])
endif

if get_option('enable-nnstreamer-backbone') and get_option('platform') != 'android'
  extra_defines += '-DENABLE_NNSTREAMER_BACKBONE=1'
endif

tflite_dep = dummy_dep

if get_option('platform') != 'android'
  tflite_dep = dependency('tensorflow2-lite', required: false)
else
  if get_option('enable-tflite-backbone') or get_option('enable-tflite-interpreter')
    tflite_version = '2.18.0'
    message('preparing tflite ' + tflite_version + ', because either tflite backbone or interpreter is enabled')
    run_command(join_paths(meson.current_source_dir(), 'jni', 'prepare_tflite.sh'), tflite_version, meson.current_build_dir(), check: true)
    tflite_root = join_paths(meson.current_build_dir(), 'tensorflow-' + tflite_version, 'tensorflow-lite')
    tflite_dep = declare_dependency(include_directories: [ join_paths('tensorflow-' + tflite_version, 'tensorflow-lite', 'include') ])
  endif
endif

if get_option('enable-tflite-backbone')
  extra_defines += '-DENABLE_TFLITE_BACKBONE=1'
endif

if get_option('enable-tflite-interpreter')
  extra_defines += '-DENABLE_TFLITE_INTERPRETER=1'
endif

if get_option('enable-onnx-interpreter')
  extra_defines += '-DENABLE_ONNX_INTERPRETER=1'
endif

opencv_dep = dummy_dep

if get_option('platform') != 'android'
  opencv_dep = dependency('opencv', required: false)
  if not opencv_dep.found()
    opencv_dep = dependency('opencv4', required: false)
    if not opencv_dep.found()
      opencv_dep = dependency('opencv3', required: false)
    endif
  endif
  if opencv_dep.found()
    extra_defines += '-DENABLE_DATA_AUGMENTATION_OPENCV=1'
  endif
endif
flatc_prog = find_program('flatc', required: false)

# Install .pc
configure_file(input: 'nntrainer.pc.in', output: 'nntrainer.pc',
  install_dir: nntrainer_libdir / 'pkgconfig',
  configuration: nntrainer_conf
)

# Install conf
configure_file(
  input: 'nntrainer.ini.in',
  output: 'nntrainer.ini',
  install_dir: nntrainer_confdir,
  configuration: nntrainer_conf
)
nntrainer_conf_abs_path = get_option('prefix') / nntrainer_confdir / 'nntrainer.ini'
message('NNTRAINER_CONF_PATH=@0@'.format(nntrainer_conf_abs_path))

if get_option('platform') != 'android'
  extra_defines += '-DNNTRAINER_CONF_PATH="@0@"'.format(nntrainer_conf_abs_path)
endif

message('extra defines are:' + ' '.join(extra_defines))
foreach defs: extra_defines
  add_project_arguments(defs, language: ['c', 'cpp'])
endforeach

# Build nntrainer
subdir('nntrainer')


# Copy libraries from CMake third party to nntrainer directory
nntrainer_dir_absolute = meson.current_build_dir() / 'nntrainer'

if get_option('platform') != 'android' and get_option('platform') != 'tizen'
  if use_custom_googletest
    if host_machine.system() == 'windows'
      message('Copy googletest libraries')
      googletest_lib_dir_absolute_win = googletest_lib_dir_absolute.replace('/', '\\')
      run_command(['xcopy', '/C', '/Y', googletest_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    endif
  endif

  if use_custom_iniparser
    if host_machine.system() == 'windows'
      message('Copy iniparser libraries')
      iniparser_lib_dir_absolute_win = iniparser_lib_dir_absolute.replace('/', '\\')
      run_command(['xcopy', '/C', '/Y', iniparser_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    endif
  endif

  if get_option('enable-opencl')
    message('Copy CLBlast libraries')
    if host_machine.system() == 'windows'
      clblast_lib_dir_absolute_win = clblast_lib_dir_absolute.replace('/', '\\')
      run_command(['xcopy', '/C', '/Y', clblast_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    else
      run_command(['cp', '-f', clblast_lib_dir_absolute / 'libclblast.so', nntrainer_dir_absolute], check: true)
      run_command(['cp', '-f', clblast_lib_dir_absolute / 'libclblast.so.1', nntrainer_dir_absolute], check: true)
      run_command(['cp', '-f', clblast_lib_dir_absolute / 'libclblast.so.1.6.3', nntrainer_dir_absolute], check: true)
    endif
  endif

  if get_option('enable-ggml')
    message('Copy ggml libraries')

    if host_machine.system() == 'windows'
      ggml_lib_dir_absolute_win = ggml_lib_dir_absolute.replace('/', '\\')
      run_command(['xcopy', '/C', '/Y', ggml_lib_dir_absolute_win, meson.current_build_dir()], check: true)
    else
      run_command(['cp', '-f', ggml_lib , nntrainer_dir_absolute], check: true)
      run_command(['cp', '-f', ggml_base_lib , nntrainer_dir_absolute], check: true)
      run_command(['cp', '-f', ggml_cpu_lib , nntrainer_dir_absolute], check: true)
    endif

    if cxx_compiler_id == 'msvc'
      ggml_dll_dir_absolute_win = ggml_dll_dir_absolute.replace('/', '\\')
      run_command(['xcopy', '/C', '/Y', ggml_dll_dir_absolute_win, meson.current_build_dir()], check: true)
    endif
  endif
endif

enable_capi = false
enable_ccapi = false
# Build api
subdir('api')

if get_option('enable-test')
  if get_option('platform') == 'android'
    warning('test is not supported in android build, test skipped')
  else
    if gtest_dep.found()
      subdir('test')
    else
      error('test enabled but gtest not found')
    endif
  endif
endif

if get_option('enable-app')
  if get_option('platform') == 'android'
    warning('android app is not supported for now, building app skipped')
  else
    if host_machine.system() != 'windows'
      if not tflite_dep.found()
        error('Tensorflow-Lite dependency not found')
      endif
    endif
    subdir('Applications')
  endif
endif

if get_option('platform') != 'android' and host_machine.system() != 'windows' and get_option('enable-nnstreamer-backbone')
  nnstreamer_dep = dependency('nnstreamer')
  message('building nnstreamer')
  subdir('nnstreamer')
else
  warning('android nnstreamer-filter and nnstreamer-trainer are not yet supported, building them is skipped')
endif

if get_option('platform') == 'android'
  subdir('jni')
endif

if get_option('platform') != 'none'
  message('building for ' + get_option('platform'))
endif

if get_option('enable-benchmarks')
  subdir('benchmarks')
endif
