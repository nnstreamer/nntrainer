lora_sources = [ 'main.cpp', cifar_path / 'cifar_dataloader.cpp' ]

  lora_dependencies = [ app_utils_dep, nntrainer_ccapi_dep ]

  e = executable('nntrainer_lora', lora_sources, include_directories
                 : [ include_directories('.'), cifar_include_dir ], dependencies
                 : lora_dependencies, install
                 : get_option('install-app'), install_dir
                 : application_install_dir)
