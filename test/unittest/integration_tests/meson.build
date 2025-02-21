test_name = 'integration_tests'
mixed_precision_test_name = 'integration_test_mixed_precision'

test_target = [
    'integration_tests.cpp',
    'integration_test_loss.cpp',
]

mixed_precision_targets = [
    model_util_path / 'models_test_utils.cpp',
    'integration_test_mixed_precision.cpp',
]

fsu_targets = [
    model_util_path / 'models_test_utils.cpp',
    'integration_test_fsu.cpp',
]

if get_option('enable-fp16')
    test_target += mixed_precision_targets
    test_target += fsu_targets
endif

exe = executable(
    test_name,
    test_target,
    include_directories: [include_directories('.'), model_util_include_dir],
    dependencies: [
        nntrainer_test_deps,
        nntrainer_dep,
        nntrainer_ccapi_dep,
    ],
    install: get_option('enable-test'),
    install_dir: application_install_dir
)

test(test_name, exe,
     args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name),
     timeout: test_timeout,
     suite: 'unittests'
)
