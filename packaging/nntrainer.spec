%define         enable_cblas 1
%define         use_gym 0
%define		nntrainerapplicationdir	%{_libdir}/nntrainer/bin
%define         test_script $(pwd)/packaging/run_unittests.sh

Name:		nntrainer
Summary:	Software framework for traning neural networks
Version:	0.0.1
Release:	0
Packager:	Jijoong Moon <jijoong.moon@sansumg.com>
License:	Apache-2.0
Source0:	nntrainer-%{version}.tar.gz
Source1001:	nntrainer.manifest

BuildRequires:	meson >= 0.50.0
BuildRequires:	openblas-devel
BuildRequires:	iniparser-devel
BuildRequires:	gtest-devel

# OpenAI interface

%define use_gym_option -Duse_gym=false

%if 0%{?use_gym}
BuildRequires:	gym-http-api-devel
%define use_gym_option -Duse_gym=true
%endif

Requires:	iniparser
Requires:	libopenblas

%description
NNtrainer is Software Framework for Training Nerual Network Models on Devices.

%package devel
Summary:	Development package for custom nntrainer developers
Requires:	nntrainer = %{version}-%{release}
Requires:	iniparser-devel
Requires:	openblas-devel

%description devel
Development pacage for custom nntrainer developers.
This contains corresponding header files and .pc pkgconfig file.

%package applications
Summary:	NNTrainer Examples
Requires:	nntrainer = %{version}-%{release}
Requires:	iniparser
BuildRequires:	tensorflow-lite-devel
BuildRequires:	pkgconfig(jsoncpp)
BuildRequires:	pkgconfig(libcurl)
BuildRequires:	pkgconfig(dlog)

%description applications
NNTraier Exmaples for test purpose.

# Using cblas for Matrix calculation
%if 0%{?enable_cblas}
%define enable_cblas -DUSE_BLAS=ON
%endif

%prep
%setup -q
cp %{SOURCE1001} .

%build
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-std=gnu++11||"`

mkdir -p build
meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} \
      --libdir=%{_libdir} --bindir=%{nntrainerapplicationdir} --includedir=%{_includedir}\
      -Dinstall-app=true -Denable-tizen=true %{use_gym_option} build

ninja -C build %{?_smp_mflags}

%if 0%{?unit_test}
bash %{test_script} ./test
%endif

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install

%files
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/*.so

%files devel
%{_includedir}/nntrainer/databuffer.h
%{_includedir}/nntrainer/layers.h
%{_includedir}/nntrainer/neuralnet.h
%{_includedir}/nntrainer/tensor.h
%{_includedir}/nntrainer/nntrainer.h
%{_includedir}/nntrainer/nntrainer_log.h
%{_includedir}/nntrainer/nntrainer_logger.h
%{_includedir}/nntrainer/optimizer.h
%{_includedir}/nntrainer/util_func.h
%{_libdir}/*.a
%{_libdir}/pkgconfig/nntrainer.pc

%files applications
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/nntrainer/bin/applications/*

%changelog
* Wed Mar 18 2020 Jijoong Moon <jijoong.moon@samsung.com>
- packaged nntrainer
