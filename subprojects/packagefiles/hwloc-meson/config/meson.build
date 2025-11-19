
# keyval for reading VERSION
keyval = import('keyval')

version_dict = keyval.load(_version_file)

# NB: meson is not turing-complete thus we can do only bounded number of substition
# Param below will be required for deeper level of substitions.
# For example config below:
#
# val=value
# foo=${val}
# bar=${foo}
#
# will require _max_ver_subst set to 2
# (worst case, depending on order of dict enumeration)
_max_ver_subst = 1 # sufficient for one level of substition.
#unsubstituted_version = false

debug(f'Performing VERSION variable substition @_max_ver_subst@ times')
foreach _ver_subst_lvl : range(_max_ver_subst)
  _subst_vers_vars = {}
  foreach k : version_dict.keys()
    v = version_dict[k]
    foreach subst_k : version_dict.keys()
      if v.contains(f'${@subst_k@}')
        repl_v = version_dict[subst_k]
        debug(f'[@_ver_subst_lvl@] Pattern is `@v@`')
        debug(f'[@_ver_subst_lvl@]   Need replace `@subst_k@` in `@k@` with `@repl_v@`')
        v = v.replace(f'${@subst_k@}', repl_v)
      endif
    endforeach
    prev_v = version_dict[k]
    if (_max_ver_subst == _ver_subst_lvl+1) and v.contains('${')
      debug(f'[@_ver_subst_lvl]Unsubstituted VERSION value for `@k@` : `@v@`')
    endif
    if prev_v != v
      debug(f'[@_ver_subst_lvl@] New val of `@k@` is @v@')
    endif
    _subst_vers_vars += {k : v.strip('"')}
  endforeach
  version_dict = _subst_vers_vars
endforeach

debug(meson.project_version())
debug(version_dict)

if version_dict['snapshot'].to_int() != 0
  assert(version_dict['snapshot_version'] == meson.project_version())
endif

#version_cfg.set10('snapshot', version_dict['snapshot'] == '1')

version_cfg.set_quoted('HWLOC_VERSION', meson.project_version(),
                       description: 'The library version, always available, even in embedded mode, contrary to VERSION')


_ver_query = {}
# Keys args to config/hwloc_get_version.sh
_defined_ver_queries = {
  'major' : 'The library version major number',
  'minor' : 'The library version minor number',
  'release' : 'The library version release number',
  'greek' : 'The library version optional greek suffix string',
}

# Collect result config/hwloc_get_version.sh
foreach query : (_defined_ver_queries.keys() + ['snapshot', 'release-date'])
  debug(f'running config/hwloc_get_version --@query@')
  ret = run_command(_get_version_script, _version_file, f'--@query@',
                    check: true,
                    capture:true)
  _ver_query += {query.underscorify(): ret.stdout().strip()}
endforeach

# VERIFY and set configuration data
foreach query : _defined_ver_queries.keys()
  k = query.underscorify() # for dicts
  k_caps = k.to_upper() # for define

  assert(_ver_query[k] == version_dict[k])

  if k not in ['greek'] # Numeric version (major, minor, release)
    version_cfg.set('HWLOC_VERSION_@k_caps@', _ver_query[k].to_int(),
                    description: _defined_ver_queries[k])
  else 
    version_cfg.set_quoted('HWLOC_VERSION_@k_caps@', _ver_query[k],
                           description: _defined_ver_queries[k])
  endif
endforeach

# FIXME: where it is used exactly beyond status?
message(_ver_query['release_date'], version_dict['date'])

# Shared library version
libhwloc_so_version = version_dict['libhwloc_so_version'].split(':')
assert(libhwloc_so_version.length() == 3)

libhwloc_so_version_current = libhwloc_so_version[0].to_int()
libhwloc_so_version_age = libhwloc_so_version[2].to_int()
libhwloc_so_name= libhwloc_so_version_current - libhwloc_so_version_age

hwloc_config_data = configuration_data()

foreach attr : ['aligned', 'always_inline', 'cold', 'const', 'deprecated', 'constructor',
                'format', 'hot', 'malloc', 'nonnull', 'noreturn', 'packed', 'pure',
                'sentinel', 'unused', 'warn_unused_result', ]
  if host_cc.has_function_attribute(attr)
    attr_upper = attr.to_upper()
    hwloc_config_data.set10(f'HWLOC_HAVE_ATTRIBUTE_@attr_upper@', true)
  endif
endforeach

# 'may_alias', 'no_instrument_function', 'weak_alias' not supported by meson

if host_cc.compiles('''
  struct { int i; } __attribute__ ((__may_alias__)) * p_value;
''', name: 'attribute may_alias')
  hwloc_config_data.set10('HWLOC_HAVE_ATTRIBUTE_MAY_ALIAS', true)
endif

if host_cc.compiles('''
int * foo(int arg1) __attribute__ ((__no_instrument_function__));
''', name: 'attribute no_instrument_function')
  hwloc_config_data.set10('HWLOC_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION', true)
endif

if host_cc.compiles('''
int foo(int arg);
int foo(int arg) { return arg + 3; }
int foo2(int arg) __attribute__ ((__weak__, __alias__("foo")));
''', name: 'attribute weak_alias')
  hwloc_config_data.set10('HWLOC_HAVE_ATTRIBUTE_WEAK_ALIAS', true)
endif

hwloc_config_data.set10('HWLOC_DEBUG', get_option('enable_debug'),
                        description: 'Whether we are in debugging mode or not')

default_hwloc_symbol_prefix = 'hwloc_'
hwloc_symbol_prefix = get_option('symbol_prefix')
if hwloc_symbol_prefix == ''
  hwloc_symbol_prefix = default_hwloc_symbol_prefix
endif

hwloc_config_data.set('HWLOC_SYM_PREFIX', hwloc_symbol_prefix,
                      description: 'The hwloc symbol prefix')
hwloc_config_data.set('HWLOC_SYM_PREFIX_CAPS', hwloc_symbol_prefix.to_upper(),
                      description: 'The hwloc symbol prefix in all caps')

hwloc_config_data.set10('HWLOC_SYM_TRANSFORM', hwloc_symbol_prefix != default_hwloc_symbol_prefix,
                        description: 'Whether we need to re-define all the hwloc public symbols or not')

hwloc_config_data.set('SIZEOF_VOID_P', host_cc.sizeof('void*'))

hwloc_config_data.set('SIZEOF_UNSIGNED_LONG', host_cc.sizeof('unsigned long'))
                      

hwloc_config_data.set('HWLOC_SIZEOF_UNSIGNED_LONG', host_cc.sizeof('unsigned long'),
                      description: 'The size of `unsigned long\', as computed by sizeof')

hwloc_config_data.set('SIZEOF_UNSIGNED_INT', host_cc.sizeof('unsigned int'))

hwloc_config_data.set('HWLOC_SIZEOF_UNSIGNED_INT', host_cc.sizeof('unsigned int'),
                      description: 'The size of `unsigned int\', as computed by sizeof')

message('Checking which OS support to include')


# host_triple is (arch, system, kernel)
host_triple = [
  host_machine.cpu_family(),
  host_machine.system(),
  host_machine.kernel()
]

os_component = []

#
# Check OS support
#
if host_triple[0] == 'ppc64' and host_triple[1] == 'bgq' and host_triple[2] == 'linux'
  hwloc_config_data.set10('HWLOC_BGQ_SYS', true,
                          description: 'Define to 1 on BlueGene/Q')
  hwloc_bgq = true
elif host_triple[2] == 'linux'
  hwloc_config_data.set10('HWLOC_LINUX_SYS', true,
                          description: 'Define to 1 on Linux')
  hwloc_linux_sys = true
  # Enable GNU/Linux extensions
  hwloc_config_data.set('_GNU_SOURCE', 1)
  if get_option('io')
    hwloc_config_data.set10('HWLOC_HAVE_LINUXIO', true)
    if not get_option('pci').disabled()
      hwloc_config_data.set10('HWLOC_HAVE_LINUXPCI', true,
                              description: 'Define to 1 if enabling Linux-specific PCI discovery in the Linux I/O component')
    endif
  endif
elif host_triple[2] == 'irix'
  # TODO: check (meson unsupported)
  hwloc_config_data.set10('HWLOC_IRIX_SYS', true,
                          description: 'Define to 1 on Irix')
elif host_triple[1] == 'darwin' # TODO: check (meson supported)
  # TODO: check (meson supported)
  # https://mesonbuild.com/Reference-tables.html#operating-system-names
  # https://mesonbuild.com/Reference-tables.html#kernel-names-since-120
  hwloc_config_data.set10('HWLOC_DARWIN_SYS', true,
                          description: 'Define to 1 on Darwin')
 
elif host_triple[1] == 'sunos' or (host_triple[2] in ['illumos', 'solaris'])
  hwloc_solaris = true
  # TODO: check (meson supported)
  # https://mesonbuild.com/Reference-tables.html#operating-system-names
  # https://mesonbuild.com/Reference-tables.html#kernel-names-since-120
  hwloc_config_data.set10('HWLOC_SOLARIS_SYS', true,
                          description: 'Define to 1 on Solaris')
elif host_triple[1] == 'aix'
  # TODO: check (not supported by meson)
  hwloc_config_data.set10('HWLOC_AIX_SYS', true,
                          description: 'Define to 1 on AIX')
elif host_triple[1] == 'hpux'
  # TODO: check (not supported by meson)
  hwloc_config_data.set10('HWLOC_HPUX_SYS', true,
                          description: 'Define to 1 on HP-UX')
  
elif host_triple[1] in ['windows', 'cygwin']
  hwloc_windows = true
  # TODO: check (meson supported)
  # https://mesonbuild.com/Reference-tables.html#operating-system-names
  # https://mesonbuild.com/Reference-tables.html#kernel-names-since-120
  hwloc_config_data.set10('HWLOC_WIN_SYS', true,
                          description: 'Define to 1 on Windows')
elif host_tiple[1] == 'freebsd'
  # TODO: check (meson supported)
  # https://mesonbuild.com/Reference-tables.html#operating-system-names
  # https://mesonbuild.com/Reference-tables.html#kernel-names-since-120
  hwloc_config_data.set10('HWLOC_FREEBSD_SYS', true,
                          description: 'Define to 1 on *FREEBSD')
elif host_triple[1] == 'netbsd' 
  # TODO: check (meson supported)
  # https://mesonbuild.com/Reference-tables.html#operating-system-names
  # https://mesonbuild.com/Reference-tables.html#kernel-names-since-120
  hwloc_config_data.set10('HWLOC_NETBSD_SYS', true,
                          description: 'Define to 1 on *NETBSD')
else
  _unsup_sys = [host_machine.cpu_family(),
                host_machine.cpu(),
                host_machine.system(),
                host_machine.kernel()]
  warning('''Unsupported! @0@
***********************************************************])
*** hwloc does not support this system.])
*** hwloc will *attempt* to build (but it may not work).])
*** hwloc run-time results may be reduced to showing just one processor,])
*** and binding will not be supported.])
*** You have been warned.])
*** Pausing to give you time to read this message...])
***********************************************************])'''.format(_unsup_sys.join('-')))
  hwloc_config_data.set10('HWLOC_UNSUPPORTED_SYS', true,
                          description: 'Define to 1 on unsupported systems')
endif

if host_machine.cpu_family() == 'x86'
  hwloc_config_data.set10('HWLOC_X86_32_ARCH', true,
                          description: 'Define to 1 on x86_32')
  hwloc_config_data.set('HWLOC_MS_LIB_ARCH', 'X86')
elif host_machine.cpu_family() == 'x86_64'
  hwloc_config_data.set10('HWLOC_X86_64_ARCH', true,
                          description: 'Define to 1 on x86_64')
  hwloc_config_data.set('HWLOC_MS_LIB_ARCH', 'X64')
endif

# Set cc_vendor variable to the name of the compiler vendor for the compiler
# for the current language.  Language must be
# one of C, OBJC, or C++.
#
# This mimics behavious of _HWLOC_CHECK_COMPILER_VENDOR
# Note: meson might not allow all of those if it'll fail it's own checks, but it is hard to verify.
# 
# thanks to http://predef.sourceforge.net/precomp.html for the list
# of defines to check.

cc_vendor = ''

# Maps meson compiler id to vendor
cc_id_to_vendor_map = {
  'gcc': 'gnu',
  'clang': 'clang', 
  'intel': 'intel',
  'intel-llvm': 'clang',
  # Since those are meant to be MSVC compatible mark the vendor as microsoft
  'clang-cl' : 'microsoft',
  'msvc': 'microsoft',
  'intel-llvm-cl': 'microsoft'
}

message('Determining host compiler vendor')
if host_cc.has_define('__ANDROID__')
  # Android
  # TODO: This seems odd, it is mostl likely clang
  cc_vendor = 'android'
elif host_cc.has_define('__INTEL_COMPILER') or host_cc.has_define('__ICC')
  # Portland Group
  cc_vendor = 'portland group'
elif host_cc.has_define('__TURBOC__')
  # Borland Turbo C
  cc_vendor = 'borland'
elif host_cc.has_define('__BORLANDC__')
  # Borland C++
  cc_vendor = 'borland'
elif host_cc.has_define('__COMO__')
  # Comeau C++
  cc_vendor = 'comeau'
elif host_cc.has_define('__DECC') or host_cc.has_define('VAXC') or \
     host_cc.has_define('__VAXC')
  # Compaq C/C++
  cc_vendor = 'compaq'
  # OSF part actually not needed anymore but doesn't hurt
  if host_cc.has_define('__osf__') and host_cc.has_define('__LANGUAGE_C__')
    if host_cc.has_define('__DECCXX') 
      # cc_vendor = 'compaq' already set
    endif
  endif
elif host_cc.has_define('_CRAYC')
  # Cray C/C++
  cc_vendor = 'cray'
elif host_cc.has_define('__DCC__')
  # Diab C/C++
  cc_vendor = 'diab'
elif host_cc.has_define('__DMC__') or host_cc.has_define('__SC__') or \
     host_cc.has_define('__ZTC__')
  # Digital Mars
  cc_vendor = 'digital mars'
elif host_cc.has_define('__HP_cc') or host_cc.has_define('__HP_aCC')
  # HP ANSI C / aC++
  cc_vendor = 'hp'
elif host_cc.has_define('__xlC__') or host_cc.has_define('__xlC__') or \
     host_cc.has_define('__IBMCPP__') or host_cc.has_define('__ibmxl__')
  # IBM XL C/C++
  cc_vendor = 'ibm'
elif host_cc.has_define('kai')
  # KAI C++ (rest in peace)
  cc_vendor = 'kai'
elif host_cc.has_define('__LCC__')
  # LCC
  cc_vendor = 'lcc'
elif host_cc.has_define('__HIGHC__')
  # MetaWare High C/C++
  cc_vendor = 'metaware high'
elif host_cc.has_define('sgi') or host_cc.has_define('__sgi')
  # MIPSpro (SGI)
  cc_vendor = 'sgi'
elif host_cc.has_define('__MRC__') or host_cc.has_define('__MRC__') or \
     host_cc.has_define('MPW_CPLUS')
  # MPW C++
  cc_vendor = 'mpw'
elif host_cc.has_define('__CC_NORCROFT')
  # Norcroft C
  cc_vendor == 'norcroft'
elif host_cc.has_define('__POCC__')
  # Pelles C
  cc_vendor == 'pelles'
elif host_cc.has_define('__SASC') or host_cc.has_define('__SASC__') or \
     host_cc.has_define('SASC')
  # SAS/C
  cc_vendor == 'sas'
elif host_cc.has_define('__SUNPRO_C') or host_cc.has_define('__SUNPRO_CC') # 
  # Sun Workshop C/C++
  cc_vendor == 'sun'
elif host_cc.has_define('__TenDRA__')
  # TenDRA C/C++
  cc_vendor = 'tendra'
elif host_cc.has_define('__TINYC__')
  # Tiny C
  cc_vendor = 'tiny'
elif host_cc.has_define('__USLC__')
  # USL C
  cc_vendor = 'usl'
elif (host_cc.get_id() in cc_id_to_vendor_map.keys())
  cc_vendor = cc_id_to_vendor_map[host_cc.get_id()]
endif

if cc_vendor == ''
  cc_vendor = 'unknown'
endif

hwloc_args_check = 0

if not host_cc.compiles('''
extern int one_arg(int x)
int foo(void) { return one_arg(1, 2); }
''', name: 'whether the C compiler accepts function calls with too many arguments')
  hwloc_args_check += 1
endif

if not host_cc.compiles('''
extern int two_arg(int x, int y);
int foo(void) { return two_arg(3); }
''', name: 'whether the C compiler accepts function calls with too few arguments')
  hwloc_args_check += 1
endif

hwloc_strict_host_cc_args = []

if hwloc_args_check != 2
  warning('Your C compiler does not consider incorrect argument counts to be a fatal error.')
  # Treated as fatal if vendor was wrongly detected and compiler doesnt accept flag
  if host_cc.has_argument('-qhalt=e', required: cc_vendor == 'ibm')
    hwloc_strict_host_cc_args += '-qhalt=e'
  elif host_cc.has_argument('-we140', required: cc_vendor == 'intel')
    hwloc_strict_host_cc_args += '-we140'
  else
    warning('Please report this warning and configure using a different C compiler if possible')
  endif
  if hwloc_strict_host_cc_args.length() > 0
    warning('Will append \'hwloc_strict_host_cc_args\' to the value  ofargs for the host C compiler when needed')
    warning('Alternatively you may configure with a different compiler.')
  endif
endif

if (hwloc_strict_host_cc_args.length() == 0) and (hwloc_args_check != 2)
  hwloc_host_cc_incorrect_args = true
endif

hwloc_want_picky = false
hwloc_host_cc_picky_args = []

# meson has support native for '-Wall' and '-Wextra' via builtin option
# if this option was not specified it will be added to build_target overrides
# when in picky mode
hwloc_picky_targets_overrides = []

if not meson.is_subproject()
  if cc_vendor == 'gnu' and fs.is_dir(meson.project_source_root()/'.git') and \
     get_option('picky').allowed()
    hwloc_want_picky = true
  endif

  if get_option('picky').enabled()
    if cc_vendor == 'gnu'
      hwloc_want_picky = true
    else
      warning('The `picky\' option was enabled, but is currently only defined for the GCC compiler set -- automatically disabled')
      hwloc_want_picky = false
    endif
  endif

  if hwloc_want_picky
    hwloc_picky_targets_overrides += 'warning_level=3'
    hwloc_host_cc_picky_args += [
      '-Wunused-parameter', '-Wundef', '-Wno-long-long', '-Wsign-compare',
      '-Wmissing-declarations', '-Wmissing-prototypes', '-Wstrict-prototypes',
      '-Wcomment', '-pedantic', '-Wshadow', '-Wwrite-strings', '-Wnested-externs',
      '-Wpointer-arith', '-Wbad-function-cast', '-Wold-style-definition',
      '-Werror-implicit-function-declaration'
    ]

    supported = host_cc.get_supported_arguments(hwloc_host_cc_picky_args, checked: 'warn')
    if supported.length() != hwloc_host_cc_picky_args.length()
      warning(f'Unsupported GCC arguments\'. Please report!')
      hwloc_host_cc_picky_args = supported
    endif

    hwloc_host_cc_picky_args += host_cc.get_supported_arguments([
      '-Wdiscarded-qualifiers', '-Wvariadic-macros', '-Wtype-limits', '-Wstack-usage=262144',
    ], checked: 'off')

    # -Wextra enables some -Wfoo that we want to disable it at some place
    if host_cc.has_argument('-Wmissing-field-initializers')
      hwloc_config_data.set10('HWLOC_HAVE_GCC_W_MISSING_FIELD_INITIALIZERS', true,
        description: 'Define to 1 if gcc -Wmissing-field-initializers is supported and enabled')
      hwloc_host_cc_picky_args += '-Wmissing-field-initializers'
    endif
    if host_cc.has_argument('-Wcast-function-type')
      hwloc_config_data.set10('_HAVE_GCC_W_CAST_FUNCTION_TYPE', true,
        description: 'Define to 1 if gcc -Wcast-function-type is supported and enabled')
      hwloc_host_cc_picky_args += '-Wcast-function-type'
    endif
    add_project_arguments(hwloc_host_cc_picky_args, language: 'c', native: false)
  endif
endif

# Now detect support

foreach hdr : ['unistd.h', 'dirent.h', 'strings.h', 'ctype.h', 'sys/mman.h']
  if host_cc.check_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                            description: f'Define 1 if has @hdr@ header file')
  endif
endforeach

foreach fn : ['strcasecmp', 'strncasecmp']
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HAVE_@0@'.format(fn_upper), true)
    hwloc_config_data.set10('HWLOC_HAVE_DECL_@0@'.format(fn_upper), true)
  endif
endforeach

foreach fn : ['strftime', 'setlocale']
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HAVE_@0@'.format(fn_upper), true)
  endif
endforeach

foreach hdr : ['stdint.h']
  if host_cc.check_header(hdr)
     hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                             description: f'Define 1 if has @hdr@ header file')
     hwloc_config_data.set10('HWLOC_HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                             description: f'Define 1 if has @hdr@ header file')
  endif
endforeach

if host_machine.system() == 'freebsd'
  message('FreeBSD-specific checks')
  foreach hdr : ['sys/domainset.h', 'sys/thr.h', 'pthread_np.h']
    if host_cc.check_header(hdr)
       hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                               description: f'Define 1 if has @hdr@ header file')
    endif
  endforeach
  
  hdr = 'sys/cpu_set.h'
  if host_cc.has_header(hdr, prefix: '#include <sys/param.h>')
     hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                             description: f'Define 1 if has @hdr@ header file')
  endif

  foreach fn : ['cpuset_setaffinity', 'cpuset_setid']
    if host_cc.has_function(fn)
      fn_upper = fn.underscorify().to_upper()
      hwloc_config_data.set10('HAVE_@0@'.format(fn_upper), true)
    endif
  endforeach
  message('end of of FreeBSD-specific checks')
endif

gdi32_dep = dependency('', required: false)
user32_dep = dependency('', required: false)
hwloc_have_user32 = false

if get_variable('hwloc_windows', false)
  foreach hdr : ['windows.h']
    if host_cc.check_header(hdr)
       hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                               description: f'Define 1 if has @hdr@ header file')
       hwloc_config_data.set10('HWLOC_HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                               description: f'Define 1 if has @hdr@ header file')
    endif
  endforeach

  win_types_to_check = [
    'KAFFINITY',
    'PROCESSOR_CACHE_TYPE',
    'CACHE_DESCRIPTOR',
    'LOGICAL_PROCESSOR_RELATIONSHIP',
    'RelationProcessorPackage',
    'RelationProcessorDie',
    'GROUP_AFFINITY',
    'PROCESSOR_RELATIONSHIP',
    'NUMA_NODE_RELATIONSHIP',
    'CACHE_RELATIONSHIP',
    'PROCESSOR_GROUP_INFO',
    'GROUP_RELATIONSHIP',
    'SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX',
    'PSAPI_WORKING_SET_EX_BLOCK',
    'PSAPI_WORKING_SET_EX_INFORMATION',
    'PROCESSOR_NUMBER',
  ]
  foreach t : win_types_to_check
    has_t = host_cc.has_type(t, prefix: '#include <windows.h>', args: '-D_WIN32_WINNT=0x0601')
    if has_t
      t_upper = t.underscorify().to_upper()
      hwloc_config_data.set10('HAVE_@0@'.format(t_upper), true)
    endif
  endforeach

  foreach fn : ['GetModuleFileName']
    if host_cc.has_function(fn, prefix: '#include <windows.h>')
      fn_upper = fn.underscorify().to_upper()
      hwloc_config_data.set10('HAVE_@0@'.format(fn_upper), true)
    endif
  endforeach

  _gdi32 = host_cc.find_library('gdi32', required: false)
  if _gdi32.found()
    gdi32_dep = _gdi32
    hwloc_config_data.set10('HAVE_LIBGDI32', true)
  endif

  _user32 = host_cc.find_library('user32', required: false)
  if _user32.found()
    user32_dep = _user32
    hwloc_config_data.set10('HWLOC_HAVE_USER32', true)
  endif
  message('end of Windows-specific checks')
endif

lgrp_dep = dependency('', required: false)
kstat_dep = dependency('', required: false)
picl_dep = dependency('', required: false)

if get_variable('hwloc_solaris', false)
  message('Solaris-specific checks')

  if host_cc.has_header('sys/lgrp_user.h')
    _llgrp = host_cc.find_library('lgrp', required: false)
    if _llgrp.found()
      fn = 'lgrp_latency_cookie'
      if host_cc.has_function(fn,
                              dependencies: _llgrp,
                              prefix: '#include <sys/lgrp_user.h>')
        lgrp_dep = _llgrp
        hwloc_config_data.set10('HAVE_LIBLGRP', true,
                                description: 'Define to 1 if we have -llgrp')
      endif
    endif
  endif

  if host_cc.has_header('kstat.h')
    _lkstat = host_cc.find_library('kstat', required: false)
    if _lkstat.found()
      kstat_dep = _lkstat
      hwloc_config_data.set10('HAVE_LIBKSTAT', true,
                              description: 'Define to 1 if we have -lkstat')
    endif
  endif

  if host_cc.has_header('picl.h')
    _lpicl = host_cc.find_library('picl', required: false)
    if _lpicl.found()
      fn = 'picl_initialize'
      if host_cc.has_function(fn,
                              dependencies: _lpicl,
                              prefix: '#include <picl.h>')
        hwloc_config_data.set10('HAVE_LIBPICL', true,
                                description: 'Define to 1 if we have -lpicl')
      endif
    endif
  endif
  message('end of Solaris-specific checks')
endif

_dep_threads = dependency('threads', required: false)

if host_machine.system() == 'aix'
  message('AIX-specific checks')
  _lpthread = host_cc.find_library('pthread', required: false)
  if _lpthread.found() or _dep_threads.found()
    
    hwloc_config_data.set10('HAVE_LIBPTHREAD', true)
    fn = 'pthread_getthrds_np'
    if host_cc.has_function(fn,
                            prefix: '#include <pthread.h>',
                            dependencies: [_lpthread, _dep_threads])
      fn_upper = fn.underscorify().to_upper()
      hwloc_config_data.set10('HAVE_@0@'.format(fn_upper), true,
                              description: 'Define to 1 if you have pthread_getthrds_np')
    endif
  endif
  message('end of AIX-specific checks')
endif

fwk_foundation_dep = dependency('', required: false)
fwk_iokit_dep = dependency('', required: false)

if host_machine.system() == 'darwin'
  message('Darwin-specific checks')

  debug('Checking for the Foundation framework')
  _fwk_foundation = dependency('appleframeworks',
                               method: 'extraframework',
                               modules: 'Foundation',
                               required: false)

  if _fwk_foundation.found()
    if host_cc.links('''
#include <CoreFoundation/CoreFoundation.h>

int main() { return CFDictionaryGetTypeID(); }

''', dependencies: _fwk_foundation)
      hwloc_config_data.set10('HWLOC_HAVE_DARWIN_FOUNDATION', true,
        description: 'Define to 1 if you have the Foundation Darwin framework')
      fwk_foundation_dep = _fwk_foundation
    endif
  endif

  debug('Checking for the IOKit framework')
  _fwk_iokit = dependency('appleframeworks',
    method: 'extraframework',
    modules: 'IOKit',
    required: false)
  if _fwk_iokit.found()
    if host_cc.links('''
#include <IOKit/IOKitLib.h>

int main() { return IORegistryGetRootEntry(kIOMasterPortDefault); }
''', dependencies: _fwk_iokit)
      hwloc_config_data.set10('HWLOC_HAVE_DARWIN_FOUNDATION', true,
        description: 'Define to 1 if you have the Foundation Darwin framework')
      fwk_iokit_dep = _fwk_iokit
     endif
   endif
   
  message('end of Darwin-specific checks')
endif

_dep_libudev = dependency('', required: false)

if (not get_variable('hwloc_bgq', false)) and host_machine.kernel() == 'linux'
  message('Linux-specific checks')
  fn = 'sched_getcpu'
  if host_cc.has_function(fn,
   prefix: '''
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <sched.h>
''')
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(fn_upper), true)
  endif
  fn = 'sched_setaffinity'
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HWLOC_HAVE_DECL_@0@'.format(fn_upper), true,
      description: f'Define to 1 if glibc provides a prototype of @fn@()')
    if get_variable('hwloc_host_cc_incorrect_args', false)
      warning('''Support for sched_setaffinity() requires a C compiler which
considers incorrect argument counts to be a fatal error.
''')
      error('Cannot continue')
    endif
    if host_cc.compiles('''
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <sched.h>
static unsigned long mask;

int main() {
 sched_setaffinity(0, (void*) &mask);
}

''', args: hwloc_strict_host_cc_args,
name: 'glibc provides the old prototype (without length) of sched_setaffinity()')
       hwloc_config_data.set10('HWLOC_HAVE_OLD_SCHED_SETAFFINITY', true,
         description: 'Define to 1 if glibc provides the old prototype (without length) of sched_setaffinity()')
     endif
  endif
  if host_cc.compiles('''
#include <sched.h>
cpu_set_t set;

''', name: 'CPU_SET works')
    hwloc_config_data.set10('HWLOC_HAVE_CPU_SET', true,
      description: 'Define to 1 if the CPU_SET macro works')
  endif
  if host_cc.compiles('''
#include <sched.h>
cpu_set_t *set;

int main() {
  set = CPU_ALLOC(1024);
  CPU_ZERO_S(CPU_ALLOC_SIZE(1024), set);
  CPU_SET_S(CPU_ALLOC_SIZE(1024), 0, set);
  CPU_FREE(set);
}
''', name: 'CPU_SET_S works')
    hwloc_config_data.set10('HWLOC_HAVE_CPU_SET_S', true,
      description: 'Define to 1 if the CPU_SET_S macro works')
  endif
  if host_cc.compiles('''
#include <unistd.h>
#include <sys/syscall.h>

int main() {
  (void)syscall(0, 1, 2, 3, 4, 5, 6);
}
''', name: 'syscall works with 6 parameters')
    hwloc_config_data.set10('HWLOC_HAVE_SYSCALL', true,
      description: 'Define to 1 if function `syscall\' is available with 6 parameters')
  endif
  _dep_libudev = dependency('libudev', required: get_option('libudev'))
  if get_option('libudev') and _dep_libudev.found()
    if host_cc.has_header('libudev.h', dependencies: _dep_libudev, required: true)
      hwloc_config_data.set10('HAVE_LIBUDEV_H', true)
    endif
    hwloc_config_data.set10('HWLOC_HAVE_LIBUDEV', true)
  endif
  message('end of Linux-specific checks')
endif

if hwloc_config_data.get('HAVE_DECL_SCHED_SETAFFINITY', 0) == 1
  hwloc_config_data.set10('HWLOC_HAVE_SCHED_SETAFFINITY', true)
endif

if host_machine.kernel() != 'linux'
  # Don't look for sys/sysctl.h on Linux because it's deprecated and
  # generates a warning in GCC10. Also it's unneeded.
  hdr = 'sys/param.h'
  sys_param_h_args = []
  if host_cc.has_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true)
    sys_param_h_args += ['-DHAVE_SYS_PARAM_H=1']
  endif

  hdr = 'sys/sysctl.h'
  if host_cc.has_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true)
    decls = [
      'CTL_HW', 'HW_NCPU', 'HW_REALMEM64', 'HW_MEMSIZE64', 'HW_PHYSMEM64',
      'HW_USERMEM64', 'HW_REALMEM', 'HW_MEMSIZE', 'HW_PHYSMEM', 'HW_USERMEM'
    ]
    foreach dcl : decls
      if host_cc.has_header_symbol('sys/sysctl.h', dcl,
        prefix: '''
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
''', args: sys_param_h_args)
        hwloc_config_data.set10('HAVE_DECL_@0@'.format(dcl.underscorify().to_upper()))
      endif
    endforeach
  endif
  # Don't detect sysctl* on Linux because its sysctl() syscall is
  # long deprecated and unneeded. Some libc still expose the symbol
  # and raise a big warning at link time.

  # Do a full link test instead of just using host_cc.has_function(), which
  # just checks to see if the symbol exists or not.  For example,
  # the prototype of sysctl uses u_int, which on some platforms
  # (such as FreeBSD) is only defined under __BSD_VISIBLE, __USE_BSD
  # or other similar definitions.  So while the symbols "sysctl" and
  # "sysctlbyname" might still be available in libc (which autoconf
  # checks for), they might not be actually usable.
  if host_cc.links('''
#include <stdio.h>
#include <sys/types.h>
#include <sys/sysctl.h>

int main() {
  return sysctl(NULL,0,NULL,NULL,NULL,0);
}
''', name: 'sysctl links')
    hwloc_config_data.set10('HAVE_SYSCTL', true,
      description: 'Define to \'1\' if sysctl is present and usable')
  endif
  if host_cc.links('''
#include <stdio.h>
#include <sys/types.h>
#include <sys/sysctl.h>

int main() {
  return sysctlbyname(NULL,NULL,NULL,NULL,0);
}
''', name: 'sysctlbyname is available')
    hwloc_config_data.set10('HAVE_SYSCTLBYNAME', true)
  endif

endif

foreach hdr : ['pthread_np.h']
  if host_cc.check_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true)
  endif
endforeach

pthread_np_h_args = []
if hwloc_config_data.get('HAVE_PTHREAD_NP_H', 0) == 1
   pthread_np_h_args += ['-DHAVE_PTHREAD_NP_H=1']
endif

foreach fn : ['pthread_setaffinity_np', 'pthread_getaffinity_np']
  if host_cc.has_function(fn ,dependencies: [_dep_threads], args: pthread_np_h_args,
     prefix: '''
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <pthread.h>
#ifdef HAVE_PTHREAD_NP_H
#  include <pthread_np.h>
#endif
''')
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(fn_upper), true)  
  endif
endforeach

# libm - fabsf and 
hwloc_new_libm = false

_lm = host_cc.find_library('m', required: false)

foreach fn : ['fabsf', 'modff']
  fn_upper = fn.underscorify().to_upper()
  if host_cc.has_function(fn, prefix: '#include <math.h>')
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(fn_upper), true)
  elif host_cc.has_function(fn, dependencies: _lm, prefix: '#include <math.h>')
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(fn_upper), true)
    hwloc_new_libm = hwloc_new_libm or true
  endif
endforeach

foreach decl : ['_SC_NPROCESSORS_ONLN', '_SC_NPROCESSORS_CONF', '_SC_NPROC_ONLN',
                '_SC_NPROC_CONF', '_SC_PAGESIZE', '_SC_PAGE_SIZE', '_SC_LARGE_PAGESIZE']
  if host_cc.has_define(decl, prefix: '#include <unistd.h>')
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(decl.underscorify().to_upper()), true)
  endif
endforeach

foreach hdr : ['mach/mach_init.h', 'mach_init.h']
  if host_cc.check_header(hdr)
     hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                             description: f'Define 1 if has @hdr@ header file')
  endif
endforeach

foreach hdr : ['mach/host_info.h']
  if host_cc.check_header(hdr)
     hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                             description: f'Define 1 if has @hdr@ header file')
     if host_cc.has_function('host_info', prefix: f'#include <@hdr@>')
       fn_upper = fn.underscorify().to_upper()
       hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
      endif
  endif
endforeach

foreach fn : ['strtoull']
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
    hwloc_config_data.set10(f'HAVE_DECL_@fn_upper@', true)
  endif
endforeach

foreach t : ['ssize_t']
  if host_cc.has_type(t, prefix: '#include <sys/types.h>')
    hwloc_config_data.set10('HAVE_@0@'.format(t.underscorify().to_upper()), true)
  endif
endforeach

foreach fn : ['snprintf', '_strdup', '_putenv']
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10('HAVE_DECL_@0@'.format(fn_upper), true)
  endif
endforeach

broken_snprintf = 'no'
if meson.is_cross_build()
  broken_snprintf = 'maybe'
  requires_run_snprintf = false
else
  requires_run_snprintf = true
endif

if host_cc.run('''
#include <stdio.h>
#include <string.h>
#include <assert.h>
int main() {
  char buf[7];
  assert(snprintf(buf, 7, "abcdef") == 6);
  assert(snprintf(buf, 6, "abcdef") == 6);
  assert(snprintf(buf, 5, "abcdef") == 6);
  assert(snprintf(buf, 0, "abcdef") == 6);
  assert(snprintf(NULL, 0, "abcdef") == 6);
  return 0;
}
''',
  required: requires_run_snprintf  
).returncode() != 0
  broken_snprintf= 'yes'
endif

if broken_snprintf == 'no'
  hwloc_config_data.set10('HWLOC_HAVE_CORRECT_SNPRINTF', true)
endif

foreach fn : ['getprogname', 'getexecname']
  has_fn = host_cc.has_function(fn)
  fn_upper = fn.underscorify().to_upper()
  hwloc_config_data.set10(f'HAVE_DECL_@fn_upper@', has_fn)
endforeach

if host_cc.links('''
  #ifndef _GNU_SOURCE
  # define _GNU_SOURCE
  #endif
  #include <errno.h>
  #include <stdio.h>
  extern char *program_invocation_name;

  int main() {return printf("%s\n", program_invocation_name);}
''',
  name: 'program_invocation_name'
)
  hwloc_config_data.set10('HAVE_PROGRAM_INVOCATION_NAME', true)
endif

if host_cc.links('''
  #include <stdio.h>
  extern char *__progname;

  int main() { return printf("%s\n", __progname); }
''',
  name: '__progname')
endif

if host_triple[1] in ['cygwin']
  hloc_pid_t = 'HANDLE'
  hwloc_thread_t = 'HANDLE'
else
  hwloc_pid_t = 'pid_t'
  tp = 'pthread_t'
  if host_cc.has_type(tp,
                      dependencies: _dep_threads,
                      prefix: '#include <pthread.h>')
    hwloc_thread_t = tp
    hwloc_config_data.set10('HAVE_@0@'.format(tp.underscorify().to_upper()), true)
  endif
endif

hwloc_config_data.set('hwloc_pid_t', hwloc_pid_t)

if is_variable('hwloc_thread_t')
  hwloc_config_data.set('hwloc_thread_t', get_variable('hwloc_thread_t'))
endif

fn = 'ffs'
_test_not_builtin = f'''
  int main() {
    int x = 0;
    return @0@(x);
   }
'''.format(fn)

if host_cc.has_function(fn) and host_cc.links(_test_not_builtin, name: f'not only __builtin_@fn@')
  _i32_builtin = fn

  fn_upper = fn.underscorify().to_upper()
  hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
  hwloc_config_data.set10(f'HWLOC_HAVE_@fn_upper@', true)
  hwloc_config_data.set10(f'HWLOC_HAVE_DECL_@fn_upper@', true)
  # FIXME: we're not doing this check for meson
  # ORIGINAL check and comment
    # May be broken due to
    # https://forums.oracle.com/forums/thread.jspa?threadID=1997328
    #  TODO: a more selective test, since bug may be version dependent.
    #  We can't use AC_TRY_LINK because the failure does not appear until
    #  run/load time and there is currently no precedent for AC_TRY_RUN
    #  use in hwloc.  --PHH
    #  For now, we're going with "all gccfss compilers are broken".
    #  Better to be safe and correct; it's not like this is
    #  performance-critical code, after all.
    # AC_DEFINE([HWLOC_HAVE_BROKEN_FFS], [1],
    #           [Define to 1 if your `ffs' function is known to be broken.])
endif

fn = 'ffsl'
foreach fn : ['ffsl', 'fls', 'flsl', 'clz', 'clzl']
  if fn.endswith('l')
    _test_not_builtin = '''
      int main() {
        long x = 0;
        return @0@(x);
       }
    '''.format(fn)
  else
    _test_not_builtin = '''
      int main() {
        int x = 0;
        return @0@(x);
       }
    '''.format(fn)
  endif
  if host_cc.has_function(fn) and host_cc.links(_test_not_builtin)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
    hwloc_config_data.set10(f'HWLOC_HAVE_@fn_upper@', true)
    hwloc_config_data.set10(f'HWLOC_HAVE_DECL_@fn_upper@', true)
  endif
endforeach

foreach hdr : ['malloc.h']
  if host_cc.has_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                            description: f'Define 1 if has @hdr@ header file')
  endif
endforeach

foreach fn : ['getpagesize', 'memalign', 'posix_memalign']
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
  endif
endforeach

foreach hdr : ['sys/utsname.h']
  if host_cc.has_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true,
                            description: f'Define 1 if has @hdr@ header file')
  endif
endforeach
foreach fn : ['uname']
  if host_cc.has_function(fn, prefix: '#include <sys/utsname.h>')
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
  endif
endforeach

foreach fn : ['pthread_mutex_lock']
  if host_cc.has_function(fn, dependencies: [_dep_threads])
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
  endif
endforeach

if hwloc_config_data.get('HAVE_PTHREAD_MUTEX_LOCK', 0) == 1
  hwloc_config_data.set10('HWLOC_HAVE_PTHREAD_MUTEX', true)
elif get_variable('hwloc_windows', false)
  error('''pthread_mutex_lock not available, required for thread-safe initialization on non-Windows platforms.
Please report this to the hwloc-users mailing list or at https://github.com/open-mpi/hwloc
Cannot continue''')
endif

non_android_fns = []
if cc_vendor != 'android'
  non_android_fns += 'openat'
endif

foreach fn : non_android_fns
  if host_cc.has_function(fn)
    fn_upper = fn.underscorify().to_upper()
    hwloc_config_data.set10(f'HAVE_@fn_upper@', true)
  endif
endforeach


# TODO: valgrind

if get_variable('hwloc_linux_sys', false) and get_option('io')
  _linuxio_enabled = true
  hwloc_config_data.set10('HWLOC_HAVE_LINUXIO', true)
else
  _linuxio_enabled = false
endif

if _linuxio_enabled and get_option('pci').allowed()
  _dep_libpciaccess = dependency('pciaccess',
    required: get_option('pci'),
    method: 'pkg-config',
    native: false)

  if _dep_libpciaccess.found()
    if host_cc.has_function('pci_slot_match_iterator_create',
                            dependencies: _dep_libpciaccess)
      _hwloc_pciaccess_ok = true
    endif
  endif
else
  _dep_libpciaccess = dependency('', required: false)
endif



_dep_libxml2 = get_option('libxml2').allowed()? \
  dependency('libxml-2.0', required: get_option('libxml2'), native: false) \
  : dependency('', required: false)

_dep_cairo = get_option('cairo').disabled()? \
  dependency('', required: false) : dependency('cairo', required: get_option('cairo'), native: false)

hwloc_all_components = [
  'pci', 'opencl', 'cuda', 'nvml', 'rsmi', 'levelzero', 'gl', 'libxml2', 'cpuid',
]

hwloc_static_components = []

foreach c : get_option('plugins')
  assert(c in hwloc_all_components)
endforeach

foreach c : hwloc_all_components
  _feat_c = get_option(c)
  if _feat_c.enabled() and (c in get_option('plugins'))
    message(f'plugin @c@')
  elif not _feat_c.disabled()
    message(f'static @c@')
  else
    message(f'not enabled @c@')
  endif
endforeach

_dep_x11 = dependency('', required: false)

if not meson.is_subproject()
  hdr = 'X11/Xlib.h'
  if host_cc.check_header(hdr)
    hwloc_config_data.set10('HAVE_@0@'.format(hdr.underscorify().to_upper()), true)
    _dep_try_x11 = dependency('X11', required: false)
    if _dep_try_x11.found()
      fn = 'XOpenDisplay'
      if host_cc.has_function(fn, dependencies: _dep_x11)
        hwloc_config_data.set10('HAVE_LIBX11', true)
        hdr = 'X11/Xutil.h'
        if host_cc.check_header(hdr, dependencies: _dep_x11)
        endif
        hdr = 'X11/keysym.h'
        if host_cc.check_header(hdr, dependencies: _dep_x11)
          
        endif
      endif
    endif
  endif 
endif

#if get_option('symbol_prefix') != ''
#endif


tst_config = configuration_data()
tst_config.merge_from(version_cfg)
tst_config.merge_from(hwloc_config_data)

_private_hwloc_config_h = configure_file(configuration: hwloc_config_data,
                                         output: 'tst-config.h')

