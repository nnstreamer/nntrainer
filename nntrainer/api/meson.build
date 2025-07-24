if get_option('enable-ccapi')
  enable_ccapi = true
  subdir('ccapi')
endif

if get_option('enable-capi').enabled()
  if not get_option('enable-ccapi')
    error('enable-ccapi must be set for capi as well')
  endif
  enable_capi = true
  subdir('capi')
elif get_option('enable-capi').auto() and get_option('enable-ccapi') and nnstreamer_capi_dep.found()
  enable_capi = true
  subdir('capi')
endif
