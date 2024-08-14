build_root = meson.build_root()

sources = ['benchmark_resnet.cpp',
           cifar_path / 'cifar_dataloader.cpp']

resnet_dependencies = [app_utils_dep,
                       iniparser_dep,
                       nntrainer_dep,
                       nntrainer_ccapi_dep,
                       benchmark_dep, ]

executable('Benchmark_ResNet',
           sources,
           include_directories : [include_directories('.'), cifar_include_dir],
           dependencies : resnet_dependencies)