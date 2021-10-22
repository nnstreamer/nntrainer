and_conf = configuration_data()

and_conf.set('MESON_CFLAGS', ' '.join(extra_defines))
and_conf.set('MESON_INIPARSER_ROOT', iniparser_root)
and_conf.set('MESON_NNTRAINER_SRCS', ' '.join(nntrainer_sources))
and_conf.set('MESON_NNTRAINER_INCS', ' '.join(nntrainer_inc_abs))

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

configure_file(input: 'Application.mk', output: 'Application.mk',
  copy: true
)

ndk_build = find_program('ndk-build', required : true)
ndk_result = custom_target('ndk-build',
                           output: 'libs',
                           build_by_default: true,
                           command: [ndk_build])

