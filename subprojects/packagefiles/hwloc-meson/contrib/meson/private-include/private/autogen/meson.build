
_config_h_in = files('config.h.in')

_autogen_cdata = configuration_data()
_autogen_cdata.merge_from(version_cfg)
_autogen_cdata.merge_from(hwloc_config_data)

_private_autogen_config_h = configure_file(
  input:  _config_h_in,
  output: 'config.h',
  configuration: _autogen_cdata,
  
)
