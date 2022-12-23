project('nntrainer', 'c', 'cpp',
  version: '0.4.0',
  license: ['apache-2.0'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)
extra_defines = ['-DMIN_CPP_VERSION=201703L']

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

if get_option('platform') == 'tizen'
  # Pass __TIZEN__ to the compiler
  add_project_arguments('-D__TIZEN__=1', language:['c','cpp'])
  add_project_arguments('-DTIZENVERSION=@0@'.format(get_option('tizen-version-major')), language: ['c', 'cpp'])
  add_project_arguments('-DTIZENVERSIONMINOR=@0@'.format(get_option('tizen-version-minor')), language: ['c', 'cpp'])

  if get_option('enable-tizen-feature-check')
    add_project_arguments('-D__FEATURE_CHECK_SUPPORT__', language: ['c', 'cpp'])
  endif
endif

warning_flags = [
  '-Wredundant-decls',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Winit-self',
  '-Waddress',
  '-Wno-multichar',
  '-Wvla',
  '-Wpointer-arith',
  '-Wno-error=varargs',
  '-Wdefaulted-function-deleted',
  '-ftree-vectorize',
  '-Wno-unused-variable'
]

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

# Set install path
nntrainer_prefix = get_option('prefix')
if get_option('platform') != 'android'
  nntrainer_libdir = nntrainer_prefix / get_option('libdir')
  nntrainer_bindir = nntrainer_prefix / get_option('bindir')
  nntrainer_includedir = nntrainer_prefix / get_option('includedir') / 'nntrainer'
  nntrainer_confdir = get_option('sysconfdir')
  application_install_dir = nntrainer_bindir / 'applications'
  nntrainer_swapdir = '/tmp'
else
  nntrainer_prefix = meson.build_root() / 'android_build_result'
  # @todo arch has to be option
  nntrainer_libdir = nntrainer_prefix / 'lib'
  nntrainer_includedir = nntrainer_prefix / 'include' / 'nntrainer'
  nntrainer_bindir = nntrainer_prefix / 'bin'
  nntrainer_confdir = nntrainer_prefix / 'conf'
  application_install_dir = nntrainer_prefix / 'examples'
  nntrainer_swapdir = '/data/local/tmp'
endif

# handle swap options
if get_option('enable-memory-swap')
  nntrainer_enable_swap = 'true'
else
  nntrainer_enable_swap = 'false'
endif

if get_option('memory-swap-path') != ''
  nntrainer_swapdir = get_option('memory-swap-path')
endif

# handle resources
nntrainer_resdir = meson.build_root() / 'res'
run_command('mkdir', '-p', nntrainer_resdir)

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
nntrainer_conf.set('MEMORY_SWAP', nntrainer_enable_swap)
nntrainer_conf.set('MEMORY_SWAP_PATH', nntrainer_swapdir)

dummy_dep = dependency('', required: false)
found_dummy_dep = declare_dependency() # dummy dep to use if found

# if ml-api-support is disabled, enable dummy common api interfaces and disable related dependencies.
ml_api_common_dep = dependency(get_option('capi-ml-common-actual'), required : get_option('ml-api-support').enabled())
nnstreamer_capi_dep = dummy_dep
if (ml_api_common_dep.found())
  nntrainer_conf.set('CAPI_ML_COMMON_DEP', get_option('capi-ml-common-actual'))
  extra_defines += '-DML_API_COMMON=1'

  nnstreamer_capi_dep = dependency(get_option('capi-ml-inference-actual'), required : true)
  extra_defines += '-DNNSTREAMER_AVAILABLE=1'
  # accessing this variable when dep_.not_found() remains hard error on purpose
  supported_nnstreamer_capi = nnstreamer_capi_dep.version().version_compare('>=1.7.0')
  if not supported_nnstreamer_capi
    extra_defines += '-DUNSUPPORTED_NNSTREAMER=1'
    warning('capi-nnstreamer version is too old, we do not know if it works with older nnstreamer version')
  endif
else
  nntrainer_conf.set('CAPI_ML_COMMON_DEP', '')
  extra_defines += '-DML_API_COMMON=0'
endif
blas_dep = dummy_dep
# Dependencies
if get_option('enable-cublas')
  extra_defines += '-DUSE_CUBLAS=1'
endif

if get_option('enable-blas')
  extra_defines += '-DUSE_BLAS=1'

  if get_option('platform') == 'android'
    message('preparing blas')
    run_command(meson.source_root() / 'jni' / 'prepare_openblas.sh', meson.build_root(), check: true)
    blas_dep = found_dummy_dep
    blas_root = meson.build_root() / 'openblas'
  else
    blas_dep = dependency('openblas')
  endif

  if blas_dep.found()
    if get_option('openblas-num-threads') > 0
      extra_defines += '-DBLAS_NUM_THREADS=@0@'.format(get_option('openblas-num-threads'))
      message('set openblas num threads=@0@'.format(get_option('openblas-num-threads')))
    endif
  endif
endif

extra_defines += '-DNNTR_NUM_THREADS=@0@'.format(get_option('nntr-num-threads'))
message('set nntrainer num threads=@0@'.format(get_option('nntr-num-threads')))

openmp_dep = dummy_dep
if get_option('enable-openmp')
  openmp_dep = dependency('openmp')
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

gmock_dep = dependency('gmock', static: true, main: false, required: false)
gtest_dep = dependency('gtest', static: true, main: false, required: false)
gtest_main_dep = dependency('gtest', static: true, main: true, required: false)


if get_option('enable-test') # and get_option('platform') != 'android'
  extra_defines += '-DENABLE_TEST=1'
  if gtest_dep.version().version_compare('<1.10.0')
     extra_defines += '-DGTEST_BACKPORT=1'
  endif
endif

if get_option('reduce-tolerance')
  extra_defines += '-DREDUCE_TOLERANCE=1'
endif

libm_dep = cxx.find_library('m') # cmath library
libdl_dep = cxx.find_library('dl') # DL library
thread_dep = dependency('threads') # pthread for tensorflow-lite

iniparser_dep = dependency('iniparser', required : false, version : '>=4.1') # iniparser
if get_option('platform') == 'android'
  message('preparing iniparser')
  run_command(meson.source_root() / 'jni' / 'prepare_iniparser.sh', meson.build_root(), check: true)
  iniparser_root = meson.build_root() / 'iniparser'
  iniparser_dep = found_dummy_dep
endif

if not iniparser_dep.found()
  message('falling back to find libiniparser library and header files')
  libiniparser_dep = cxx.find_library('iniparser')
  sysroot = run_command(
    cxx.cmd_array() + ['-print-sysroot']
    ).stdout().split('\n')[0]

  if sysroot.startswith('/')
    sysroot_inc_cflags_template = '-I@0@/usr/include@1@'
    sysroot_inc = sysroot_inc_cflags_template.format(sysroot, '')
    add_project_arguments(sysroot_inc, language: ['c', 'cpp'])
    sysroot_inc_cflags_iniparser = sysroot_inc_cflags_template.format(sysroot,
      '/iniparser')
  else
    sysroot_inc_cflags_iniparser = '-I/usr/include/iniparser'
  endif

  if libiniparser_dep.found() and cxx.has_header('iniparser.h', \
        args : sysroot_inc_cflags_iniparser)
    iniparser_dep = declare_dependency (dependencies : libiniparser_dep,
      compile_args : sysroot_inc_cflags_iniparser)
  else
    error('Failed to resolve dependency on iniparser')
  endif
endif

if get_option('platform') == 'android'
  message('preparing ml api')
  run_command(meson.source_root() / 'jni' / 'prepare_ml-api.sh', meson.build_root() / 'ml-api-inference', check: true)
  ml_api_common_root = meson.build_root() / 'ml-api-inference'
  ml_api_inc = ml_api_common_root / 'include'
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'ml-api-common.h', nntrainer_includedir)
  )
  meson.add_install_script(
    'sh', '-c', 'cp @0@ ${DESTDIR}@1@'.format(ml_api_inc / 'tizen_error.h', nntrainer_includedir)
  )
  ml_api_common_dep = found_dummy_dep
endif

if get_option('enable-nnstreamer-backbone') and get_option('platform') != 'android'
  extra_defines += '-DENABLE_NNSTREAMER_BACKBONE=1'
endif

tflite_dep = dummy_dep

if get_option('platform') != 'android'
  tflite_dep = dependency('tensorflow2-lite', required: false)
else
  if get_option('enable-tflite-backbone') or get_option('enable-tflite-interpreter')
    message('preparing tflite, because either tflite backbone or interpreter is enabled')
    run_command(meson.source_root() / 'jni' / 'prepare_tflite.sh', '2.3.0', meson.build_root(), check: true)
    tflite_root = meson.build_root() / 'tensorflow-2.3.0' / 'tensorflow-lite'
    tflite_dep = found_dummy_dep
  endif
endif

if get_option('enable-tflite-backbone')
  extra_defines += '-DENABLE_TFLITE_BACKBONE=1'
endif

if get_option('enable-tflite-interpreter')
  extra_defines += '-DENABLE_TFLITE_INTERPRETER=1'
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
    # this is needed for reinforcement application. We can move this to reinforecement app dependency
    jsoncpp_dep = dependency('jsoncpp') # jsoncpp
    libcurl_dep = dependency('libcurl')
    if not tflite_dep.found()
      error('Tensorflow-Lite dependency not found')
    endif
    subdir('Applications')
  endif
endif

if get_option('enable-nnstreamer-tensor-filter')
  if get_option('platform') == 'android'
    warning('android nnstreamer-filter is not yet supported, building nnstreamer-filter skipped')
  else
    nnstreamer_dep = dependency('nnstreamer', required: true)
    subdir('nnstreamer/tensor_filter')
  endif
endif

if get_option('platform') == 'android'
  subdir('jni')
endif

if get_option('platform') != 'none'
  message('building for ' + get_option('platform'))
endif
