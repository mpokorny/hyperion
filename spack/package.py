# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Hyperion(CMakePackage):
    """High performance radio astronomy data processing prototyping"""

    # Add a proper url for your package's homepage here.
    homepage = "https://www.nrao.edu"
    git      = "https://github.com/mpokorny/hyperion.git"

    maintainers = ['mpokorny']

    version('master', branch='master')

    generator = 'Ninja'

    # CMake build dependency
    depends_on('cmake@3.13:', type='build')
    depends_on('ninja', type='build')

    # Top-level hyperion variants
    max_dims_list = ('4','5','6','7','8','9')
    variant('max_dims', default='8', description='Maximum index space rank',
            values=max_dims_list, multi=False)
    variant('casacore', default=True, description='Enable casacore support')
    variant('hdf5', default=True, description='Enable HDF5 support')
    variant('yaml', default=True, description='Enable YAML support')
    variant('cuda', default=False, description='Enable CUDA support')
    cuda_arch_list = \
        ('30','32','35','50','52','53','60','61','62','70','72','75')
    variant('cuda_arch',
            default=('70'),
            values=cuda_arch_list,
            description='Target CUDA compute capabilities',
            multi=True)
    variant('kokkos', default=False, description='Enable Kokkos support')
    variant('openmp', default=False, description='Enable OpenMP support')
    variant('shared', default=True, description='Build shared libraries')

    # Legion dependency
    for md in max_dims_list:
        depends_on(f'legion max_dims={md}', when=f'max_dims={md}')
    depends_on('legion+redop_complex')
    depends_on('legion+hdf5', when='+hdf5')
    depends_on('legion+kokkos+shared_libs', when='+kokkos')
    depends_on('legion~kokkos', when='~kokkos')
    depends_on('legion~cuda', when='~cuda')
    for arch in cuda_arch_list:
        depends_on(
            f'legion+cuda cuda_arch={arch}',
            when=f'+cuda cuda_arch={arch}')

    # Kokkos dependency
    depends_on('kokkos+shared+serial std=17', when='+kokkos')
    depends_on('kokkos+openmp', when='+kokkos+openmp')
    # FIXME: don't require nvcc_wrapper when compiling with Clang
    depends_on('kokkos+cuda+cuda_lambda~cuda_host_init_check +wrapper',
               when='+kokkos+cuda')
    for arch in cuda_arch_list:
        depends_on(
            f'kokkos+cuda cuda_arch={arch}',
            when=f'+kokkos+cuda cuda_arch={arch}')

    # Other dependencies
    depends_on('zlib')
    depends_on('fftw~mpi precision=float,double')
    depends_on('fftw+openmp', when='+openmp')
    depends_on('casacore', when='+casacore')
    depends_on('pkgconf', when='+casacore', type=('build')) # FindCasacore requires it
    depends_on('hdf5+threadsafe+szip~mpi', when='+hdf5')
    depends_on('yaml-cpp', when='+yaml')
    depends_on('cuda', when='+cuda', type=('build', 'link', 'run'))

    # Need Python to run tests
    depends_on('python@3:', type='test')

    def cmake_args(self):
        args = []
        spec = self.spec

        args.append(self.define_from_variant('BUILD_SHARED_LIBS', 'shared'))
        args.append(self.define_from_variant('USE_CASACORE', 'casacore'))
        args.append(self.define_from_variant('USE_HDF5', 'hdf5'))
        args.append(self.define_from_variant('USE_OPENMP', 'openmp'))
        args.append(self.define_from_variant('USE_CUDA', 'cuda'))
        args.append(self.define_from_variant('USE_KOKKOS', 'kokkos'))
        args.append(self.define_from_variant('hyperion_USE_YAML', 'yaml'))

        if '+cuda' in spec:
            # TODO: this is probably not desired, but removal needs testing
            cxx_std = "14"
        else:
            cxx_std = "17"

        args.append(f'-DBUILD_ARCH:STRING={spec.architecture.target}')
        args.append(f'-Dhyperion_CXX_STANDARD={cxx_std}')
        if '+kokkos' in spec:
            args.append(f'-DCMAKE_CXX_COMPILER={self.spec["kokkos"].kokkos_cxx}')
        return args
