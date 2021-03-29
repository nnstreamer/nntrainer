if get_option('enable-ccapi')
  subdir('ccapi')
endif

if get_option('enable-capi')
  if not get_option('enable-ccapi')
    error('enable-ccapi must be set for capi as well')
  endif
  subdir('capi')
endif
