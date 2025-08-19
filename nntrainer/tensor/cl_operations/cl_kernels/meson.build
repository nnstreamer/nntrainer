fs = import('fs')

cl_op_kernels = [
  'rmsnorm.cl',
]

all_includes = ''
foreach k : cl_op_kernels
  basename = fs.stem(k)
  conf_data = configuration_data(
    {
      'namespace': 'nntrainer',
      'variable_name': basename + '_kernel',
      'header_path': basename + '.h',
      'kernel_code': fs.read(k),
    },
  )

  configure_file(
    input: 'kernel.cpp.in',
    output: basename + '.cpp',
    configuration: conf_data,
  )
  configure_file(input: 'kernel.h.in', output: basename + '.h', configuration: conf_data)

  # source.full_path() does not work: https://github.com/mesonbuild/meson/issues/5273#issuecomment-1851811417
  # can be used when meson is upgraded to 1.4.0
  nntrainer_headers += meson.current_build_dir() / basename + '.h'
  nntrainer_sources += meson.current_build_dir() / basename + '.cpp'

  all_includes += f'#include "@basename@.h"\n'
endforeach
configure_file(
  input: 'kernels.h.in',
  output: 'cl_kernels.h',
  configuration: {'includes': all_includes},
)
nntrainer_headers += meson.current_build_dir() / 'cl_kernels.h'

nntrainer_inc += include_directories('..')
nntrainer_inc_abs += meson.current_build_dir() / '..'
