
hwloc_core_srcs = []

hwloc_core_srcs += _private_autogen_config_h
hwloc_core_srcs += _autogen_config_h

hwloc_srcs = files(
  'topology.c',
  'traversal.c',
  'distances.c',
  'memattrs.c',
  'cpukinds.c',
  'components.c',
  'bind.c',
  'bitmap.c',
  'pci-common.c',
  'diff.c',
  'shmem.c',
  'misc.c',
  'base64.c',
  'topology-noos.c',
  'topology-synthetic.c',
  'topology-xml.c',
  'topology-xml-nolibxml.c',
)

hwloc_components = []

static_components_arr = ''
foreach comp : hwloc_components
  static_components_arr += '  &hwloc_${comp}_component,\n'
endforeach

_static_components_h = configure_file(
  input: _hwloc_static_components_h_in,
  output: 'static-components.h',
  configuration: { 'static_components': static_components_arr}
)

if get_variable('hwloc_linux_sys', false)
  message('lnx')
  hwloc_srcs += files('topology-linux.c', 'topology-hardwired.c')
  _hwloc_deps += _dep_libudev
  _hwloc_deps += _dep_libpciaccess
endif

library('hwloc', [hwloc_core_srcs, hwloc_srcs],
  include_directories: [
    _hwloc_inc,
    _inc_autogen,
    _inc_private_autogen
  ],
  c_args: '-DHWLOC_INSIDE_LIBHWLOC',
  dependencies: _hwloc_deps,
)
