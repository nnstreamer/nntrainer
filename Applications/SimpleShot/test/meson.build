gtest_dep_with_main = dependency('gtest', main : true, required : false)


test_target = [
  'mock_test',
]

foreach target: test_target
  exe = executable(
    target,
    target + '.cpp',
    dependencies: gtest_dep_with_main,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target, exe)
endforeach
