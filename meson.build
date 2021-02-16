project('nntrainer', 'c', 'cpp',
  version: '0.0.1',
  license: ['apache-2.0'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++14'
  ]
)
add_project_arguments('-DMIN_CPP_VERSION=201402', language:['c','cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
build_platform = ''

if get_option('enable-tizen')
  # Pass __TIZEN__ to the compiler
  build_platform = 'tizen'
  add_project_arguments('-D__TIZEN__=1', language:['c','cpp'])

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
  '-O2',
  '-ftree-vectorize'
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
nntrainer_libdir = nntrainer_prefix / get_option('libdir')
nntrainer_bindir = nntrainer_prefix / get_option('bindir')
nntrainer_includedir = nntrainer_prefix / get_option('includedir')
nntrainer_inidir = get_option('sysconfdir')
application_install_dir = nntrainer_bindir / 'applications'

# Set default configuration
nntrainer_conf = configuration_data()
nntrainer_conf.set('VERSION', meson.project_version())
nntrainer_conf.set('PREFIX', nntrainer_prefix)
nntrainer_conf.set('EXEC_PREFIX', nntrainer_bindir)
nntrainer_conf.set('LIB_INSTALL_DIR', nntrainer_libdir)
nntrainer_conf.set('INCLUDE_INSTALL_DIR', nntrainer_includedir)

dummy_dep = dependency('', required: false)

blas_dep = dummy_dep
# Dependencies
if get_option('enable-cublas')
   add_project_arguments('-DUSE_CUBLAS=1', language:['c','cpp'])
endif

if get_option('enable-blas')
  add_project_arguments('-DUSE_BLAS=1', language:['c','cpp'])
  if build_platform == 'tizen'
    blas_dep = dependency('openblas')
  else
    blas_dep = dependency('blas-openblas', required:false)
    # for Ubuntu 20.04
    if not blas_dep.found()
      blas_dep = dependency('openblas')
    endif
  endif
endif

if get_option('enable-profile')
  add_project_arguments('-DPROFILE=1', language:['c', 'cpp'])
endif

if get_option('use_gym')
   add_project_arguments('-DUSE_GYM=1', language:['c','cpp'])
endif

if get_option('enable-logging')
   add_project_arguments('-D__LOGGING__=1', language:['c','cpp'])
endif

if get_option('enable-test')
  add_project_arguments('-DENABLE_TEST=1', language:['c','cpp'])
endif

libm_dep = cxx.find_library('m') # cmath library
libdl_dep = cxx.find_library('dl') # DL library
thread_dep = dependency('threads') # pthread for tensorflow-lite
iniparser_dep = dependency('iniparser', required : false) # iniparser
if not iniparser_dep.found()
  message('falling back to find libiniparser library and header files')
  libiniparser_dep = cxx.find_library('iniparser')
  if libiniparser_dep.found() and cxx.has_header('iniparser.h', \
        args : '-I/usr/include/iniparser')
    iniparser_dep = declare_dependency (dependencies : libiniparser_dep,
      compile_args : '-I/usr/include/iniparser')
  endif
endif

nnstreamer_capi_dep = dependency('capi-ml-inference', required:false)
if nnstreamer_capi_dep.found()
  add_project_arguments('-DNNSTREAMER_AVAILABLE=1', language:['c','cpp'])
  # accessing this variable when dep_.not_found() remains hard error on purpose
  supported_nnstreamer_capi = nnstreamer_capi_dep.version().version_compare('>=1.7.0')
  if not supported_nnstreamer_capi
    add_project_arguments('-DUNSUPPORTED_NNSTREAMER=1', language:['c','cpp'])
    warning('capi-nnstreamer version is too old, we do not know if it works with older nnstreamer version')
  endif
endif

ml_api_common_dep = dependency('capi-ml-common', required:true)

if get_option('enable-nnstreamer-backbone')
  add_project_arguments('-DENABLE_NNSTREAMER_BACKBONE=1', language:['c','cpp'])
endif

tflite_dep = dependency('tensorflow-lite', required: false)
if get_option('enable-tflite-backbone')
  add_project_arguments('-DENABLE_TFLITE_BACKBONE=1', language:['c','cpp'])
endif

gtest_dep = dependency('gtest', required: false)

opencv_dep = dependency('opencv', required: false)
if not opencv_dep.found()
  opencv_dep = dependency('opencv4', required: false)
  if not opencv_dep.found()
    opencv_dep = dependency('opencv3', required: false)
  endif
endif
if opencv_dep.found()
  add_project_arguments('-DENABLE_DATA_AUGMENTATION_OPENCV=1', language:['c','cpp'])
endif

# Install .pc
configure_file(input: 'nntrainer.pc.in', output: 'nntrainer.pc',
  install_dir: nntrainer_libdir / 'pkgconfig',
  configuration: nntrainer_conf
)

# Build nntrainer
subdir('nntrainer')

# Build api
subdir('api')

if get_option('enable-app')
  jsoncpp_dep = dependency('jsoncpp') # jsoncpp
  libcurl_dep = dependency('libcurl')
  if not tflite_dep.found()
    error('Tensorflow-Lite dependency not found')
  endif
  subdir('Applications')
endif

if get_option('enable-test')
  subdir('test')
endif

if get_option('enable-nnstreamer-tensor-filter')
  nnstreamer_dep = dependency('nnstreamer', required: true)
  subdir('nnstreamer/tensor_filter')
endif

if get_option('enable-android')

  ndk_build = find_program('ndk-build', required : true)
  jni_root = meson.current_source_dir() / 'jni'
  jni_build_root = meson.current_build_dir() / 'jni'

  ndk_args = {
    'NDK_PROJECT_PATH': jni_root,
    'APP_BUILD_SCRIPT': jni_root / 'Android.mk',
    'NDK_APPLICATION_MK': jni_root / 'Application.mk',
    'NDK_LIBS_OUT': jni_build_root / 'libs',
    'NDK_OUT': jni_build_root / 'objs',
    'ENABLE_PROFILE': get_option('enable-profile') ? 1 : 0,
    'ENABLE_BLAS': get_option('enable-blas') ? 1 : 0,
  }

  num_threads = run_command('grep', '-c', '^processor', '/proc/cpuinfo').stdout().strip()
  message('num processor are: ' + num_threads)

  thread_opt_flag = '-j' + num_threads

  ndk_additional_flags = [thread_opt_flag]

  ndk_build_command = [ndk_build]
  foreach key, val : ndk_args
    ndk_build_command += '@0@=@1@'.format(key, val)
  endforeach
  ndk_build_command += ndk_additional_flags

  android_build_target = custom_target('android',
    output: 'jni',
    build_by_default: true,
    command: ndk_build_command
  )

endif
