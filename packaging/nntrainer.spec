# Execute gbs with --define "testcoverage 1" in case that you must get unittest coverage statistics
%define         enable_cblas 1
%define         use_gym 0
%define		nntrainerapplicationdir	%{_libdir}/nntrainer/bin
%define         test_script $(pwd)/packaging/run_unittests.sh
%define         gen_input $(pwd)/test/input_gen/genInput.py
%bcond_with tizen

Name:		nntrainer
Summary:	Software framework for traning neural networks
Version:	0.1.0.rc1
Release:	0
Packager:	Jijoong Moon <jijoong.moon@sansumg.com>
License:	Apache-2.0
Source0:	nntrainer-%{version}.tar.gz
Source1001:	nntrainer.manifest
%if %{with tizen}
Source1002:     capi-nntrainer.manifest
%endif
Source2001:	trainset.tar.gz
Source2002:	valset.tar.gz
Source2003:	testset.tar.gz
Source2004:	label.dat
Source2005:	unittest_layers.tar.gz

BuildRequires:	meson >= 0.50.0
BuildRequires:	openblas-devel
BuildRequires:	iniparser-devel
BuildRequires:	gtest-devel
BuildRequires:	python3
BuildRequires:	python3-numpy
BuildRequires:	capi-ml-common-devel

# OpenAI interface

%define use_gym_option -Duse_gym=false

%if 0%{?use_gym}
BuildRequires:	gym-http-api-devel
%define use_gym_option -Duse_gym=true
%endif

%if 0%{?testcoverage}
# to be compatible with gcc-9, lcov should have a higher version than 1.14.1
BuildRequires: lcov
# BuildRequires:	taos-ci-unittest-coverage-assessment
%endif

%if %{with tizen}
BuildRequires:	pkgconfig(capi-system-info)
BuildRequires:	pkgconfig(capi-base-common)
BuildRequires:	pkgconfig(dlog)
%endif  # tizen

Requires:	iniparser
Requires:	libopenblas_pthreads0

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

%package devel-static
Summary:        Static library for nntrainer-devel package
Requires:       devel = %{version}-%{release}
%description devel-static
Static library package of nntrainer-devel

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



%if 0%{?testcoverage}
%package unittest-coverage
Summary:	NNTrainer UnitTest Coverage Analysis Result

%description unittest-coverage
HTML pages of lcov results of NNTrainer generated during rpmbuild
%endif

%if %{with tizen}
%package -n capi-nntrainer
Summary:         Tizen Native API for NNTrainer
Group:           Multimedia/Framework
Requires:        %{name} = %{version}-%{release}
%description -n capi-nntrainer
Tizen Native API wrapper for NNTrainer.
You can train neural networks efficiently.

%post -n capi-nntrainer -p /sbin/ldconfig
%postun -n capi-nntrainer -p /sbin/ldconfig

%package -n capi-nntrainer-devel
Summary:         Tizen Native API Devel Kit for NNTrainer
Group:           Multimedia/Framework
Requires:        capi-nntrainer = %{version}-%{release}
Requires:        capi-ml-common-devel
%description -n capi-nntrainer-devel
Developmental kit for Tizen Native NNTrainer API.

%package -n capi-nntrainer-devel-static
Summary:         Static library for Tizen Native API
Group:           Multimedia/Framework
Requires:        capi-nntrainer-devel = %{version}-%{release}
%description -n capi-nntrainer-devel-static
Static library of capi-nntrainer-devel package.

%endif #tizen

## Define build options
%define enable_tizen -Denable-tizen=false
%define enable_tizen_feature_check -Denable-tizen-feature-check=true

%if %{with tizen}
%define enable_tizen -Denable-tizen=true
%endif

# Using cblas for Matrix calculation
%if 0%{?enable_cblas}
%define enable_cblas -DUSE_BLAS=ON
%endif

%prep
%setup -q
cp %{SOURCE1001} .
cp %{SOURCE2001} .
cp %{SOURCE2002} .
cp %{SOURCE2003} .
cp %{SOURCE2004} .
cp %{SOURCE2005} .

%if %{with tizen}
cp %{SOURCE1002} .
%endif

%build
CXXFLAGS=`echo $CXXFLAGS | sed -e "s|-std=gnu++11||"`

%if 0%{?testcoverage}
CXXFLAGS="${CXXFLAGS} -fprofile-arcs -ftest-coverage"
CFLAGS="${CFLAGS} -fprofile-arcs -ftest-coverage"
%endif

mkdir -p build
meson --buildtype=plain --prefix=%{_prefix} --sysconfdir=%{_sysconfdir} \
      --libdir=%{_libdir} --bindir=%{nntrainerapplicationdir} --includedir=%{_includedir}\
      -Dinstall-app=true %{enable_tizen} %{enable_tizen_feature_check} %{use_gym_option} build

ninja -C build %{?_smp_mflags}

%if 0%{?unit_test}
tar xzf trainset.tar.gz -C build
tar xzf valset.tar.gz -C build
tar xzf testset.tar.gz -C build
cp label.dat build
tar xzf unittest_layers.tar.gz -C build
bash %{test_script} ./test
%endif

%install
DESTDIR=%{buildroot} ninja -C build %{?_smp_mflags} install

%if 0%{?testcoverage}
##
# The included directories are:
#
# api: nnstreamer api
# gst: the nnstreamer elements
# nnstreamer_example: custom plugin examples
# common: common libraries for gst (elements)
# include: common library headers and headers for external code (packaged as "devel")
#
# Intentionally excluded directories are:
#
# tests: We are not going to show testcoverage of the test code itself or example applications

%if %{with tizen}
%define testtarget $(pwd)/api/capi
%else
%define testtarget
%endif

# 'lcov' generates the date format with UTC time zone by default. Let's replace UTC with KST.
# If you ccan get a root privilege, run ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
TZ='Asia/Seoul'; export TZ
$(pwd)/test/unittestcoverage.py module $(pwd)/nntrainer %testtarget

# Get commit info
VCS=`cat ${RPM_SOURCE_DIR}/nntrainer.spec | grep "^VCS:" | sed "s|VCS:\\W*\\(.*\\)|\\1|"`

# Create human readable unit test coverage report web page.
# Create null gcda files if gcov didn't create it because there is completely no unit test for them.
find . -name "*.gcno" -exec sh -c 'touch -a "${1%.gcno}.gcda"' _ {} \;
# Remove gcda for meaningless file (CMake's autogenerated)
find . -name "CMakeCCompilerId*.gcda" -delete
find . -name "CMakeCXXCompilerId*.gcda" -delete
#find . -path "/build/*.j

# Generate report
lcov -t 'NNTrainer Unit Test Coverage' -o unittest.info -c -d . -b %{_builddir}/%{name}-%{version}/build --include "*/nntrainer/*" --include "*/api/*" --exclude "*/tensorflow/*"

# Exclude generated files
lcov -r unittest.info "*/test/*" "*/meson*/*" -o unittest-filtered.info

# Visualize the report
genhtml -o result unittest-filtered.info -t "nntrainer %{version}-%{release} ${VCS}" --ignore-errors source -p ${RPM_BUILD_DIR}

mkdir -p %{buildroot}%{_datadir}/nntrainer/unittest/
cp -r result %{buildroot}%{_datadir}/nntrainer/unittest/
%endif  # test coverage

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/libnntrainer.so

%files devel
%{_includedir}/nntrainer/databuffer.h
%{_includedir}/nntrainer/databuffer_file.h
%{_includedir}/nntrainer/databuffer_func.h
%{_includedir}/nntrainer/layer.h
%{_includedir}/nntrainer/input_layer.h
%{_includedir}/nntrainer/fc_layer.h
%{_includedir}/nntrainer/bn_layer.h
%{_includedir}/nntrainer/conv2d_layer.h
%{_includedir}/nntrainer/pooling2d_layer.h
%{_includedir}/nntrainer/flatten_layer.h
%{_includedir}/nntrainer/loss_layer.h
%{_includedir}/nntrainer/activation_layer.h
%{_includedir}/nntrainer/neuralnet.h
%{_includedir}/nntrainer/model_loader.h
%{_includedir}/nntrainer/tensor.h
%{_includedir}/nntrainer/lazy_tensor.h
%{_includedir}/nntrainer/tensor_dim.h
%{_includedir}/nntrainer/nntrainer_log.h
%{_includedir}/nntrainer/nntrainer_logger.h
%{_includedir}/nntrainer/optimizer.h
%{_includedir}/nntrainer/util_func.h
%{_includedir}/nntrainer/parse_util.h
%{_includedir}/nntrainer/nntrainer-api-common.h
%{_libdir}/pkgconfig/nntrainer.pc

%files devel-static
%{_libdir}/*.a
%exclude %{_libdir}/libcapi*.a

%if %{with tizen}
%files -n capi-nntrainer
%manifest capi-nntrainer.manifest
%license LICENSE
%{_libdir}/libcapi-nntrainer.so

%files -n capi-nntrainer-devel
%{_includedir}/nntrainer/nntrainer.h
%{_includedir}/nntrainer/nntrainer-api-common.h
%{_libdir}/pkgconfig/capi-nntrainer.pc

%files -n capi-nntrainer-devel-static
%{_libdir}/libcapi-nntrainer.a

%endif #tizen

%files applications
%manifest nntrainer.manifest
%defattr(-,root,root,-)
%license LICENSE
%{_libdir}/nntrainer/bin/applications/*

%if 0%{?testcoverage}
%files unittest-coverage
%{_datadir}/nntrainer/unittest/*
%endif

%changelog
* Mon Aug 10 2020 Jijoong Moon <jijoong.moon@samsung.com>
- Release of 0.1.0.rc1
* Wed Mar 18 2020 Jijoong Moon <jijoong.moon@samsung.com>
- packaged nntrainer
