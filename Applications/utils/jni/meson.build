utils_inc = include_directories('./includes')

app_utils_source = [
  'bitmap_helpers.cpp'
]

shared_library('app_utils',
	app_utils_source,
	include_directories: utils_inc,
	install: get_option('install-app'),
	install_dir: application_install_dir
)

app_utils_lib = static_library('app_utils',
        app_utils_source,
	include_directories: utils_inc,
	install: get_option('install-app'),
	install_dir: application_install_dir
)

app_utils_dep = declare_dependency(
    link_with:app_utils_lib,
    include_directories: utils_inc
)
