
_include_HEADERS = []

build_standalone = not meson.is_subproject()

if (build_standalone)
  _include_HEADERS = files('hwloc.h')

  _include_hwloc_HEADERS = []
  _include_hwloc_HEADERS += files(
    'hwloc/bitmap.h',
    'hwloc/helper.h',
    'hwloc/inlines.h',
    'hwloc/memattrs.h',
    'hwloc/cpukinds.h',
    'hwloc/diff.h',
    'hwloc/shmem.h',
    'hwloc/distances.h',
    'hwloc/export.h',
    'hwloc/openfabrics-verbs.h',
    'hwloc/opencl.h',
    'hwloc/cuda.h',
    'hwloc/cudart.h',
    'hwloc/nvml.h',
    'hwloc/rsmi.h',
    'hwloc/levelzero.h',
    'hwloc/plugins.h',
    'hwloc/gl.h',
    'hwloc/rename.h',
    'hwloc/deprecated.h',
  )

  _private_HEADERS = files(
    'private/private.h',
    'private/debug.h',
    'private/misc.h',
    'private/xml.h',
    'private/components.h',
    'private/internal-components.h',
    'private/cpuid-x86.h',
    'private/windows.h',
  )
endif
