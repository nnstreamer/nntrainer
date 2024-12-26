subgraph_source = ['main.cpp', cifar_path / 'cifar_dataloader.cpp']
subgraph_dependencies = [app_utils_dep, nntrainer_dep, nntrainer_ccapi_dep]

e = executable(
    'nntrainer_subgraph',
    subgraph_source,
    include_directories: [include_directories('.'), cifar_include_dir],
    dependencies: subgraph_dependencies,
    install: get_option('install-app'),
    install_dir: application_install_dir,
)
