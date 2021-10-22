and_conf = configuration_data()

and_conf.set('MESON_CFLAGS', ' '.join(extra_defines))
and_conf.set('MESON_NNTRAINER_SRCS', ' '.join(nntrainer_sources))
and_conf.set('MESON_NNTRAINER_INCS', ' '.join(nntrainer_inc_abs))
and_conf.set('MESON_CCAPI_NNTRAINER_SRCS', ' '.join(ccapi_src))
and_conf.set('MESON_CCAPI_NNTRAINER_INCS', ' '.join(ccapi_inc_abs))
and_conf.set('MESON_CAPI_NNTRAINER_SRCS', ' '.join(capi_src))
and_conf.set('MESON_CAPI_NNTRAINER_INCS', ' '.join(capi_inc_abs))

if iniparser_dep.found()
  and_conf.set('MESON_INIPARSER_ROOT', iniparser_root)
endif

if tflite_dep.found()
  and_conf.set('MESON_HAS_TFLITE', 1)
  and_conf.set('MESON_TFLITE_ROOT', tflite_root)
endif

if blas_dep.found()
  and_conf.set('MESON_BLAS_ROOT', blas_root)
else
  error('blas is needed for the android build')
endif

if ml_api_common_dep.found()
  and_conf.set('MESON_ML_API_COMMON_ROOT', ml_api_common_root)
else
  error('ml api common dep is needed for the android build')
endif



configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)

install_data(sources: 'Android-prebuilt.mk', rename: 'Android.mk', install_dir: nntrainer_prefix)

configure_file(input: 'Application.mk', output: 'Application.mk',
  copy: true
)

outputs= [
  'libc++_shared.so',
  'libcapi-nntrainer.so',
  'libccapi-nntrainer.so',
  'libnnstreamer-native.so',
  'libnntrainer.so'
]


ndk_build = find_program('ndk-build', required : true)

ndk_args = {
  'TARGET_OUT': meson.current_build_dir(),
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

ndk_result = custom_target('ndk-build',
                           output: outputs,
                           build_by_default: true,
                           install: true,
                           install_dir: nntrainer_libdir,
                           command: [ndk_build_command])

