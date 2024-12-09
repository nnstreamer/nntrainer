build_root = meson.build_root()

sources = ['benchmark_resnet.cpp',
           fake_datagen_path / 'fake_data_gen.cpp']

resnet_dependencies = [nntrainer_dep,
                       nntrainer_ccapi_dep,
                       benchmark_dep, ]

executable('Benchmark_ResNet',
           sources,
           include_directories : [include_directories('.'), fake_datagen_include_dir],
           dependencies : resnet_dependencies)
