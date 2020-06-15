/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <hyperion/synthesis/Zernike.h>

#if !HAVE_CXX17

using namespace hyperion::synthesis;

constexpr const std::array<XYPolyTerm, 1> Zernike<0, 0>::terms;
constexpr const std::array<XYPolyTerm, 1> Zernike<-1, 1>::terms;
constexpr const std::array<XYPolyTerm, 1> Zernike<1, 1>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<-2, 2>::terms;
constexpr const std::array<XYPolyTerm, 1> Zernike<0, 2>::terms;
constexpr const std::array<XYPolyTerm, 2> Zernike<2, 2>::terms;
constexpr const std::array<XYPolyTerm, 2> Zernike<-3, 3>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<-1, 3>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<1, 3>::terms;
constexpr const std::array<XYPolyTerm, 2> Zernike<3, 3>::terms;
constexpr const std::array<XYPolyTerm, 2> Zernike<-4, 4>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<-2, 4>::terms;
constexpr const std::array<XYPolyTerm, 6> Zernike<0, 4>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<2, 4>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<4, 4>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<-5, 5>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<-3, 5>::terms;
constexpr const std::array<XYPolyTerm, 6> Zernike<-1, 5>::terms;
constexpr const std::array<XYPolyTerm, 6> Zernike<1, 5>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<3, 5>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<5, 5>::terms;
constexpr const std::array<XYPolyTerm, 3> Zernike<-6, 6>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<-4, 6>::terms;
constexpr const std::array<XYPolyTerm, 6> Zernike<-2, 6>::terms;
constexpr const std::array<XYPolyTerm, 10> Zernike<0, 6>::terms;
constexpr const std::array<XYPolyTerm, 8> Zernike<2, 6>::terms;
constexpr const std::array<XYPolyTerm, 7> Zernike<4, 6>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<6, 6>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<-7, 7>::terms;
constexpr const std::array<XYPolyTerm, 7> Zernike<-5, 7>::terms;
constexpr const std::array<XYPolyTerm, 9> Zernike<-3, 7>::terms;
constexpr const std::array<XYPolyTerm, 10> Zernike<-1, 7>::terms;
constexpr const std::array<XYPolyTerm, 10> Zernike<1, 7>::terms;
constexpr const std::array<XYPolyTerm, 9> Zernike<3, 7>::terms;
constexpr const std::array<XYPolyTerm, 7> Zernike<5, 7>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<7, 7>::terms;
constexpr const std::array<XYPolyTerm, 4> Zernike<-8, 8>::terms;
constexpr const std::array<XYPolyTerm, 7> Zernike<-6, 8>::terms;
constexpr const std::array<XYPolyTerm, 8> Zernike<-4, 8>::terms;
constexpr const std::array<XYPolyTerm, 10> Zernike<-2, 8>::terms;
constexpr const std::array<XYPolyTerm, 15> Zernike<0, 8>::terms;
constexpr const std::array<XYPolyTerm, 12> Zernike<2, 8>::terms;
constexpr const std::array<XYPolyTerm, 12> Zernike<4, 8>::terms;
constexpr const std::array<XYPolyTerm, 8> Zernike<6, 8>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<8, 8>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<-9, 9>::terms;
constexpr const std::array<XYPolyTerm, 9> Zernike<-7, 9>::terms;
constexpr const std::array<XYPolyTerm, 11> Zernike<-5, 9>::terms;
constexpr const std::array<XYPolyTerm, 13> Zernike<-3, 9>::terms;
constexpr const std::array<XYPolyTerm, 15> Zernike<-1, 9>::terms;
constexpr const std::array<XYPolyTerm, 15> Zernike<1,9>::terms;
constexpr const std::array<XYPolyTerm, 13> Zernike<3, 9>::terms;
constexpr const std::array<XYPolyTerm, 11> Zernike<5, 9>::terms;
constexpr const std::array<XYPolyTerm, 9> Zernike<7, 9>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<9, 9>::terms;
constexpr const std::array<XYPolyTerm, 5> Zernike<-10, 10>::terms;
constexpr const std::array<XYPolyTerm, 8> Zernike<-8, 10>::terms;
constexpr const std::array<XYPolyTerm, 12> Zernike<-6, 10>::terms;
constexpr const std::array<XYPolyTerm, 12> Zernike<-4, 10>::terms;
constexpr const std::array<XYPolyTerm, 15> Zernike<-2, 10>::terms;
constexpr const std::array<XYPolyTerm, 21> Zernike<0, 10>::terms;
constexpr const std::array<XYPolyTerm, 18> Zernike<2, 10>::terms;
constexpr const std::array<XYPolyTerm, 18> Zernike<4, 10>::terms;
constexpr const std::array<XYPolyTerm, 14> Zernike<6, 10>::terms;
constexpr const std::array<XYPolyTerm, 11> Zernike<8, 10>::terms;
constexpr const std::array<XYPolyTerm, 6> Zernike<10, 10>::terms;
#endif // !HAVE_CXX17

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
