%define         enable_cblas 1

Name:		nntrainer
Summary:	Software framework for traning neural networks
Version:        0.0.1
Release:        0
Packager:       Jijoong Moon <jijoong.moon@sansumg.com>
License:        Apache-2.0
Source0:        nntrainer-%{version}.tar.gz
Source1001:     nntrainer.manifest

BuildRequires:  cmake >= 2.8.3
BuildRequires:  openblas-devel
BuildRequires:  iniparser-devel
Requires:       iniparser
Requires:       libopenblas

%description
NNtrainer is Software Framework for Training Nerual Network Models on Devices.

%package devel
Summary:        Development package for custom nntrainer developers
Requires:       nntrainer = %{version}-%{release}
Requires:       iniparser-devel
Requires:       openblas-devel

%description devel
Development pacage for custom nntrainer developers.
This contains corresponding header files and .pc pkgconfig file.

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
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr %{enable_cblas} -DTIZEN=ON
make %{?jobs:-j%jobs}
popd

%install
pushd build
%make_install
popd


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
%{_libdir}/pkgconfig/nntrainer.pc

%changelog
* Wed Mar 18 2020 Jijoong Moon <jijoong.moon@samsung.com>
- packaged nntrainer
