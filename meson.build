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
nntrainer_libdir = join_paths(nntrainer_prefix, get_option('libdir'))
nntrainer_bindir = join_paths(nntrainer_prefix, get_option('bindir'))
nntrainer_includedir = join_paths(nntrainer_prefix, get_option('includedir'))
nntrainer_inidir = get_option('sysconfdir')
application_install_dir = join_paths(nntrainer_bindir, 'applications')

# Set default configuration
nntrainer_conf = configuration_data()
nntrainer_conf.set('VERSION', meson.project_version())
nntrainer_conf.set('PREFIX', nntrainer_prefix)
nntrainer_conf.set('EXEC_PREFIX', nntrainer_bindir)
nntrainer_conf.set('LIB_INSTALL_DIR', nntrainer_libdir)
nntrainer_conf.set('INCLUDE_INSTALL_DIR', nntrainer_includedir)

dummy_dep = dependency('', required: false)
openmp_dep = dependency('openmp')

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
    blas_dep = dependency('blas-openblas')
  endif
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

nnstreamer_capi_dep = dependency('capi-nnstreamer', required:false)
if get_option('enable-nnstreamer-backbone')
  add_project_arguments('-DENABLE_NNSTREAMER_BACKBONE=1', language:['c','cpp'])
endif

jsoncpp_dep = dependency('jsoncpp') # jsoncpp
libcurl_dep = dependency('libcurl')
gtest_dep = dependency('gtest', required: false)

# Install .pc
configure_file(input: 'nntrainer.pc.in', output: 'nntrainer.pc',
  install_dir: join_paths(nntrainer_libdir, 'pkgconfig'),
  configuration: nntrainer_conf
)

# Build nntrainer
subdir('nntrainer')

# Build api
subdir('api')

if get_option('enable-app')
  tflite_dep = dependency('tensorflow-lite', required: true)
  subdir('Applications')
endif

if get_option('enable-test')
   subdir('test')
endif

if get_option('enable-nnstreamer-tensor-filter')
   nnstreamer_dep = dependency('nnstreamer', required: true)
   subdir('nnstreamer/tensor_filter')
endif
