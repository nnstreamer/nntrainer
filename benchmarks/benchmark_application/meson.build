build_root = meson.build_root()

sources = ['benchmark_resnet.cpp',
           fake_datagen_path / 'fake_data_gen.cpp']

resnet_dependencies = [nntrainer_dep,
                       nntrainer_ccapi_dep,
                       benchmark_dep, ]

benchmark_ling_args = ''

if host_machine.system() == 'windows'
    benchmark_ling_args = '-lshlwapi'
endif

executable('Benchmark_ResNet',
           sources,
           include_directories : [include_directories('.'), fake_datagen_include_dir],
           dependencies : resnet_dependencies,
           link_args: benchmark_ling_args)
