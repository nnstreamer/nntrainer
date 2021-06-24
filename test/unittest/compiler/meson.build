test_target = [
  'unittest_compiler',
  # 'unittest_interpreter'
]

foreach target: test_target
  exe = executable(
    target,
    target + '.cpp',
    dependencies: nntrainer_test_main_deps,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target, exe, args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), target)] )
endforeach
