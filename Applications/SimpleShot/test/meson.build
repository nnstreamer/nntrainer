gtest_dep_with_main = dependency('gtest', main : true, required : false)


test_target = [
  'simpleshot_utils'
]

foreach target: test_target
  exe = executable(
    target,
    target + '.cpp',
    dependencies: [gtest_dep_with_main, simpleshot_test_dep],
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target, exe)
endforeach
