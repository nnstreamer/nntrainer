
input_gen_dep = [
  nntrainer_dep,
  nntrainer_ccapi_dep,
  nntrainer_testutil_dep
]

gen_target = [
  ['genModelExeOrder', []],
]

foreach target: gen_target
  exe = executable(
    target[0],
    [target[0] + '.cpp'] + target[1],
    # below is temporary measure, we will eventually remove unittest_nntrainer_models
    include_directories: nntrainer_test_inc,
    dependencies: input_gen_dep
  )
endforeach

