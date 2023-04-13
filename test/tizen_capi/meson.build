unittest_tizen_deps = [
  nntrainer_capi_dep,
  nntrainer_test_deps,
]

unittest_name_list = ['', '_layer', '_optimizer', '_lr_scheduler', '_dataset']
unittest_prefix = 'unittest_tizen_capi'

foreach test_name : unittest_name_list
  unittest_name = unittest_prefix + test_name

  exec = executable(
    unittest_name,
    unittest_name + '.cpp',
    dependencies: unittest_tizen_deps,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )

  test(unittest_name, exec, timeout: 120, args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), unittest_name))
endforeach
