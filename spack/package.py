# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install hyperion
#
# You can edit this file again by typing:
#
#     spack edit hyperion
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *


class Hyperion(CMakePackage):
    """High performance radio astronomy data processing prototyping"""

    # Add a proper url for your package's homepage here.
    homepage = "https://www.nrao.edu"
    git      = "https://github.com/mpokorny/hyperion.git"

    maintainers = ['mpokorny']

    version('master', branch='master', submodules=True)

    # Add dependencies if required.
    depends_on('cmake@3.13:', type='build')

    variant('casacore', default=True, description='Enable casacore support')
    variant('hdf5', default=True, description='Enable HDF5 support')
    variant('yaml', default=True, description='Enable YAML support')
    variant('llvm', default=False, description='Enable LLVM support')
    variant('max_dim', default='7', description='Maximum index space rank',
            values=('4','5','6','7','8','9'), multi=False)
    variant('debug', default=False, description='Enable debug flags')
    variant('lg_debug', default=False, description='Enable Legion debug flags')
    variant('lg_bounds_checks', default=False, description='Enable Legion bounds checks')
    variant('lg_privilege_checks', default=False, description='Enable Legion privilege checks')

    depends_on('zlib')
    depends_on('casacore', when='+casacore')
    depends_on('hdf5+threadsafe+szip~mpi', when='+hdf5')
    depends_on('yaml-cpp', when='+yaml')
    depends_on('llvm@6.0.1', when='+llvm')
    # not sure how to make this a dependency only for running tests
    depends_on('python@3:', type='run')

    def cmake_args(self):
        args = []
        spec = self.spec

        args.append(self.define_from_variant('USE_CASACORE', 'casacore'))
        args.append(self.define_from_variant('USE_HDF5', 'hdf5'))
        args.append(self.define_from_variant('USE_YAML', 'yaml'))
        args.append(self.define_from_variant('MAX_DIM', 'max_dim'))
        args.append(self.define_from_variant('Legion_BOUNDS_CHECKS', 'lg_bounds_checks'))
        args.append(self.define_from_variant('Legion_PRIVILEGE_CHECKS', 'lg_privilege_checks'))

        if '+debug' in spec:
            args.append('-DCMAKE_BUILD_TYPE=Debug')
        if '+lg_debug' in spec:
            args.append('-DLegion_CMAKE_BUILD_TYPE=Debug')

        return args
