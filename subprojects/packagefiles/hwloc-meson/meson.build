project('hwloc', ['c'],
  version: run_command(files('config/hwloc_get_version.sh'),
                       files('VERSION'), 
                       '--version',
                       check: true,
                       capture: true).stdout().strip(),
  meson_version: '>=1.5.0',
)

fs = import('fs')

_hwloc_runstatedir = get_option('localstatedir')/'run'
message(f'RUNSTATEDIR @_hwloc_runstatedir@')

add_project_arguments(f'-DRUNSTATEDIR="@_hwloc_runstatedir@"', language:'c', native: false)

hwloc_mode = (meson.is_subproject())? 'embedded' : 'standalone'

_version_file = files('VERSION')
_get_version_script = files('config'/'hwloc_get_version.sh')

build_cc = meson.get_compiler('c', native: true)
host_cc = meson.get_compiler('c', native: false)

_hwloc_deps = []

if false
  add_languages('cpp', native: true)
  add_languages('cpp', native: false)

  build_cxx = meson.get_compiler('cpp', native: true)
  host_cxx = meson.get_compiler('cpp', native: false)
endif

version_cfg = configuration_data()

_hwloc_inc = include_directories('include')

subdir('config')
subdir('contrib'/'meson')
subdir('hwloc')
