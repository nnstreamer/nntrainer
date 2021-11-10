if not nnstreamer_capi_dep.found()
  message('nnstreamer_capi dep not found, skipping ml_inference test')
else
  test_name = 'test_ml_inference'

  test_target = [
    'test_nnstreamer_single.cpp'
  ]

  exe = executable(
    test_name,
    test_target,
    dependencies: [
      nntrainer_test_main_deps,
      nnstreamer_capi_dep
    ],
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )

  nntrainer_filter_env = environment()
  nntrainer_filter_env.set('NNSTREAMER_FILTERS', meson.build_root() / 'nnstreamer' / 'tensor_filter')

  # inside gbs build, ml api feature is not enabled,
  # we can't run this test inside gbs until we have a workaround.
  if get_option('platform') != 'tizen'
    test(test_name, exe,
      args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name),
      env: nntrainer_filter_env
    )
  endif

endif
