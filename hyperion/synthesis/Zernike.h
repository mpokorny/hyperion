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
#ifndef HYPERION_SYNTHESIS_ZERNIKE_H_
#define HYPERION_SYNTHESIS_ZERNIKE_H_
#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#include <algorithm>
#include <array>
#include <vector>

#include <experimental/mdspan>
#ifdef HYPERION_USE_KOKKOS
# include <Kokkos_Core.hpp>
#endif

namespace hyperion {
namespace synthesis {

struct XYPolyTerm {
  unsigned px;
  unsigned py;
  int c;
};

template <int M, unsigned N>
struct Zernike {
  // template <size_t P>
  // static const array<XYPolyTerm, P> terms;
};
template <>
struct Zernike<0, 0> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 0> N;
  static const constexpr std::array<XYPolyTerm, 1> terms{
    XYPolyTerm{0, 0, 1}};
};
template <>
struct Zernike<-1, 1> {
  typedef std::integral_constant<int, -1> M;
  typedef std::integral_constant<unsigned, 1> N;
  static const constexpr std::array<XYPolyTerm, 1> terms{
    XYPolyTerm{0, 1, 1}};
};
template <>
struct Zernike<1, 1> {
  typedef std::integral_constant<int, 1> M;
  typedef std::integral_constant<unsigned, 1> N;
  static const constexpr std::array<XYPolyTerm, 1> terms{
    XYPolyTerm{1, 0, 1}};
};
template <>
struct Zernike<-2, 2> {
  typedef std::integral_constant<int, -2> M;
  typedef std::integral_constant<unsigned, 2> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{2, 0, 2},
    XYPolyTerm{0, 0, -1},
    XYPolyTerm{0, 2, 2}};
};
template <>
struct Zernike<0, 2> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 2> N;
  static const constexpr std::array<XYPolyTerm, 1> terms{
    XYPolyTerm{1, 1, 2}};
};
template <>
struct Zernike<2, 2> {
  typedef std::integral_constant<int, 2> M;
  typedef std::integral_constant<unsigned, 2> N;
  static const constexpr std::array<XYPolyTerm, 2> terms{
    XYPolyTerm{2, 0, -1},
    XYPolyTerm{0, 2, 1}};
};
template <>
struct Zernike<-3, 3> {
  typedef std::integral_constant<int, -3> M;
  typedef std::integral_constant<unsigned, 3> N;
  static const constexpr std::array<XYPolyTerm, 2> terms{
    XYPolyTerm{3, 0, -1},
    XYPolyTerm{1, 2, 3}};
};
template <>
struct Zernike<-1, 3> {
  typedef std::integral_constant<int, -1> M;
  typedef std::integral_constant<unsigned, 3> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{3, 0, 3},
    XYPolyTerm{1, 0, -2},
    XYPolyTerm{1, 2, 3}};
};
template <>
struct Zernike<1, 3> {
  typedef std::integral_constant<int, 1> M;
  typedef std::integral_constant<unsigned, 3> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{2, 1, 3},
    XYPolyTerm{0, 1, -2},
    XYPolyTerm{0, 3, 3}};
};
template <>
struct Zernike<3, 3> {
  typedef std::integral_constant<int, 3> M;
  typedef std::integral_constant<unsigned, 3> N;
  static const constexpr std::array<XYPolyTerm, 2> terms{
    XYPolyTerm{2, 1, -3},
    XYPolyTerm{0, 3, 1}};
};
template <>
struct Zernike<-4, 4> {
  typedef std::integral_constant<int, -4> M;
  typedef std::integral_constant<unsigned, 4> N;
  static const constexpr std::array<XYPolyTerm, 2> terms{
    XYPolyTerm{3, 1, -4},
    XYPolyTerm{1, 3, 4}};
};
template <>
struct Zernike<-2, 4> {
  typedef std::integral_constant<int, -2> M;
  typedef std::integral_constant<unsigned, 4> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{3, 1, 8},
    XYPolyTerm{1, 1, -6},
    XYPolyTerm{1, 3, 8}};
};
template <>
struct Zernike<0, 4> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 4> N;
  static const constexpr std::array<XYPolyTerm, 6> terms{
    XYPolyTerm{4, 0, 6},
    XYPolyTerm{2, 0, -6},
    XYPolyTerm{2, 2, 12},
    XYPolyTerm{0, 0, 1},
    XYPolyTerm{0, 2, -6},
    XYPolyTerm{0, 4, 6}};
};
template <>
struct Zernike<2, 4> {
  typedef std::integral_constant<int, 2> M;
  typedef std::integral_constant<unsigned, 4> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{4, 0, -4},
    XYPolyTerm{2, 0, 3},
    XYPolyTerm{0, 2, -3},
    XYPolyTerm{0, 4, 4}};
};
template <>
struct Zernike<4, 4> {
  typedef std::integral_constant<int, 4> M;
  typedef std::integral_constant<unsigned, 4> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{4, 0, 1},
    XYPolyTerm{2, 2, -6},
    XYPolyTerm{0, 4, 1}};
};
template <>
struct Zernike<-5, 5> {
  typedef std::integral_constant<int, -5> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{5, 0, 1},
    XYPolyTerm{3, 2, -10},
    XYPolyTerm{1, 4, 5}};
};
template <>
struct Zernike<-3, 5> {
  typedef std::integral_constant<int, -3> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{5, 0, -5},
    XYPolyTerm{3, 0, 4},
    XYPolyTerm{3, 2, 10},
    XYPolyTerm{1, 2, -12},
    XYPolyTerm{1, 4, 15}};
};
template <>
struct Zernike<-1, 5> {
  typedef std::integral_constant<int, -1> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 6> terms{
    XYPolyTerm{5, 0, 10},
    XYPolyTerm{3, 0, -12},
    XYPolyTerm{3, 2, 20},
    XYPolyTerm{1, 0, 3},
    XYPolyTerm{1, 2, -12},
    XYPolyTerm{1, 4, 10}};
};
template <>
struct Zernike<1, 5> {
  typedef std::integral_constant<int, 1> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 6> terms{
    XYPolyTerm{4, 1, 10},
    XYPolyTerm{2, 1, -12},
    XYPolyTerm{2, 3, 20},
    XYPolyTerm{0, 1, 3},
    XYPolyTerm{0, 3, -12},
    XYPolyTerm{0, 5, 10}};
};
template <>
struct Zernike<3, 5> {
  typedef std::integral_constant<int, 3> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{4, 1, -15},
    XYPolyTerm{2, 1, 12},
    XYPolyTerm{2, 3, -10},
    XYPolyTerm{0, 3, -4},
    XYPolyTerm{0, 5, 5}};
};
template <>
struct Zernike<5, 5> {
  typedef std::integral_constant<int, 5> M;
  typedef std::integral_constant<unsigned, 5> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{4, 1, 5},
    XYPolyTerm{2, 3, -10},
    XYPolyTerm{0, 5, 1}};
};
template <>
struct Zernike<-6, 6> {
  typedef std::integral_constant<int, -6> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 3> terms{
    XYPolyTerm{5, 1, 6},
    XYPolyTerm{3, 3, -20},
    XYPolyTerm{1, 5, 6}};
};
template <>
struct Zernike<-4, 6> {
  typedef std::integral_constant<int, -4> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{5, 1, -24},
    XYPolyTerm{3, 1, 20},
    XYPolyTerm{1, 3, -20},
    XYPolyTerm{1, 5, 24}};
};
template <>
struct Zernike<-2, 6> {
  typedef std::integral_constant<int, -2> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 6> terms{
    XYPolyTerm{1, 1, 12},
    XYPolyTerm{3, 1, 40},
    XYPolyTerm{1, 3, -40},
    XYPolyTerm{5, 1, 30},
    XYPolyTerm{3, 3, 60},
    XYPolyTerm{1, 5, -30}};
};
template <>
struct Zernike<0, 6> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 10> terms{
    XYPolyTerm{6, 0, 20},
    XYPolyTerm{4, 0, -30},
    XYPolyTerm{4, 2, 60},
    XYPolyTerm{2, 0, 12},
    XYPolyTerm{2, 2, -60},
    XYPolyTerm{2, 4, 60},
    XYPolyTerm{0, 6, 20},
    XYPolyTerm{0, 4, -30},
    XYPolyTerm{0, 2, 12},
    XYPolyTerm{0, 0, -1}};
};
template <>
struct Zernike<2, 6> {
  typedef std::integral_constant<int, 2> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 8> terms{
    XYPolyTerm{6, 0, -15},
    XYPolyTerm{4, 0, 20},
    XYPolyTerm{4, 2, -15},
    XYPolyTerm{2, 0, -6},
    XYPolyTerm{2, 4, 15},
    XYPolyTerm{0, 2, 6},
    XYPolyTerm{0, 4, -20},
    XYPolyTerm{0, 6, 15}};
};
template <>
struct Zernike<4, 6> {
  typedef std::integral_constant<int, 4> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 7> terms{
    XYPolyTerm{6, 0, 6},
    XYPolyTerm{4, 0, -5},
    XYPolyTerm{4, 2, -30},
    XYPolyTerm{2, 2, 30},
    XYPolyTerm{2, 4, -30},
    XYPolyTerm{0, 4, -5},
    XYPolyTerm{0, 6, 6}};
};
template <>
struct Zernike<6, 6> {
  typedef std::integral_constant<int, 6> M;
  typedef std::integral_constant<unsigned, 6> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{6, 0, -1},
    XYPolyTerm{4, 2, 15},
    XYPolyTerm{2, 4, -15},
    XYPolyTerm{0, 6, 1}};
};
template <>
struct Zernike<-7, 7> {
  typedef std::integral_constant<int, -7> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{7, 0, -1},
    XYPolyTerm{5, 2, 21},
    XYPolyTerm{3, 4, -35},
    XYPolyTerm{1, 6, 7}};
};
template <>
struct Zernike<-5, 7> {
  typedef std::integral_constant<int, -5> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 7> terms{
    XYPolyTerm{7, 0, 7},
    XYPolyTerm{5, 0, -6},
    XYPolyTerm{5, 2, -63},
    XYPolyTerm{3, 2, 60},
    XYPolyTerm{3, 4, -35},
    XYPolyTerm{1, 4, -30},
    XYPolyTerm{1, 6, 35}};
};
template <>
struct Zernike<-3, 7> {
  typedef std::integral_constant<int, -3> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 9> terms{
    XYPolyTerm{7, 0, -21},
    XYPolyTerm{5, 0, 30},
    XYPolyTerm{5, 2, 21},
    XYPolyTerm{3, 0, -10},
    XYPolyTerm{3, 2, -60},
    XYPolyTerm{3, 4, 105},
    XYPolyTerm{1, 2, 30},
    XYPolyTerm{1, 4, -90},
    XYPolyTerm{1, 6, 63}};
};
template <>
struct Zernike<-1, 7> {
  typedef std::integral_constant<int, -1> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 10> terms{
    XYPolyTerm{7, 0, 35},
    XYPolyTerm{5, 0, -60},
    XYPolyTerm{5, 2, 105},
    XYPolyTerm{3, 0, 30},
    XYPolyTerm{3, 2, -120},
    XYPolyTerm{3, 4, 105},
    XYPolyTerm{1, 0, -4},
    XYPolyTerm{1, 2, 30},
    XYPolyTerm{1, 4, -60},
    XYPolyTerm{1, 6, 35}};
};
template <>
struct Zernike<1, 7> {
  typedef std::integral_constant<int, 1> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 10> terms{
    XYPolyTerm{6, 1, 35},
    XYPolyTerm{4, 1, -60},
    XYPolyTerm{4, 3, 105},
    XYPolyTerm{2, 1, 30},
    XYPolyTerm{2, 3, -120},
    XYPolyTerm{2, 5, 105},
    XYPolyTerm{0, 1, -4},
    XYPolyTerm{0, 3, 30},
    XYPolyTerm{0, 5, -60},
    XYPolyTerm{0, 7, 35}};
};
template <>
struct Zernike<3, 7> {
  typedef std::integral_constant<int, 3> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 9> terms{
    XYPolyTerm{6, 1, -63},
    XYPolyTerm{4, 1, 90},
    XYPolyTerm{4, 3, -105},
    XYPolyTerm{2, 1, -30},
    XYPolyTerm{2, 3, 60},
    XYPolyTerm{2, 5, -21},
    XYPolyTerm{0, 3, 10},
    XYPolyTerm{0, 5, -30},
    XYPolyTerm{0, 7, 21}};
};
template <>
struct Zernike<5, 7> {
  typedef std::integral_constant<int, 5> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 7> terms{
    XYPolyTerm{6, 1, 35},
    XYPolyTerm{4, 1, -30},
    XYPolyTerm{4, 3, -35},
    XYPolyTerm{2, 3, 60},
    XYPolyTerm{2, 5, -63},
    XYPolyTerm{0, 5, -6},
    XYPolyTerm{0, 7, 7}};
};
template <>
struct Zernike<7, 7> {
  typedef std::integral_constant<int, 7> M;
  typedef std::integral_constant<unsigned, 7> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{6, 1, -7},
    XYPolyTerm{4, 3, 35},
    XYPolyTerm{2, 5, -21},
    XYPolyTerm{0, 7, 1}};
};
template <>
struct Zernike<-8, 8> {
  typedef std::integral_constant<int, -8> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 4> terms{
    XYPolyTerm{7, 1, -8},
    XYPolyTerm{5, 3, 56},
    XYPolyTerm{3, 5, -56},
    XYPolyTerm{1, 7, 8}};
};
template <>
struct Zernike<-6, 8> {
  typedef std::integral_constant<int, -6> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 7> terms{
    XYPolyTerm{7, 1, 48},
    XYPolyTerm{5, 1, -42},
    XYPolyTerm{5, 3, -112},
    XYPolyTerm{3, 3, 140},
    XYPolyTerm{3, 5, -112},
    XYPolyTerm{1, 5, -42},
    XYPolyTerm{1, 7, 48}};
};
template <>
struct Zernike<-4, 8> {
  typedef std::integral_constant<int, -4> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 8> terms{
    XYPolyTerm{7, 1, -112},
    XYPolyTerm{5, 1, 168},
    XYPolyTerm{5, 3, -112},
    XYPolyTerm{3, 1, -60},
    XYPolyTerm{3, 5, 112},
    XYPolyTerm{1, 3, 60},
    XYPolyTerm{1, 5, -168},
    XYPolyTerm{1, 7, 112}};
};
template <>
struct Zernike<-2, 8> {
  typedef std::integral_constant<int, -2> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 10> terms{
    XYPolyTerm{7, 1, -112},
    XYPolyTerm{5, 1, -210},
    XYPolyTerm{5, 3, 336},
    XYPolyTerm{3, 1, 120},
    XYPolyTerm{3, 3, -420},
    XYPolyTerm{3, 5, 336},
    XYPolyTerm{1, 1, -20},
    XYPolyTerm{1, 3, 120},
    XYPolyTerm{1, 5, -210},
    XYPolyTerm{1, 7, 112}};
};
template <>
struct Zernike<0, 8> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 15> terms{
    XYPolyTerm{8, 0, 70},
    XYPolyTerm{6, 0, -140},
    XYPolyTerm{6, 2, 280},
    XYPolyTerm{4, 0, 90},
    XYPolyTerm{4, 2, -420},
    XYPolyTerm{4, 4, 420},
    XYPolyTerm{2, 0, -20},
    XYPolyTerm{2, 2, 180},
    XYPolyTerm{2, 4, -420},
    XYPolyTerm{2, 6, 280},
    XYPolyTerm{0, 0, 1},
    XYPolyTerm{0, 2, -20},
    XYPolyTerm{0, 4, 90},
    XYPolyTerm{0, 6, -140},
    XYPolyTerm{0, 8, 70}};
};
template <>
struct Zernike<2, 8> {
  typedef std::integral_constant<int, 2> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 12> terms{
    XYPolyTerm{8, 0, -56},
    XYPolyTerm{6, 0, 105},
    XYPolyTerm{6, 2, -112},
    XYPolyTerm{4, 0, -60},
    XYPolyTerm{4, 2, 105},
    XYPolyTerm{2, 0, 10},
    XYPolyTerm{2, 4, -105},
    XYPolyTerm{2, 6, 112},
    XYPolyTerm{0, 2, -10},
    XYPolyTerm{0, 4, 60},
    XYPolyTerm{0, 6, -105},
    XYPolyTerm{0, 8, 56}};
};
template <>
struct Zernike<4, 8> {
  typedef std::integral_constant<int, 4> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 12> terms{
    XYPolyTerm{8, 0, 28},
    XYPolyTerm{6, 0, -42},
    XYPolyTerm{6, 2, -112},
    XYPolyTerm{4, 0, 15},
    XYPolyTerm{4, 2, 210},
    XYPolyTerm{4, 4, -280},
    XYPolyTerm{2, 2, -90},
    XYPolyTerm{2, 4, 210},
    XYPolyTerm{2, 6, -112},
    XYPolyTerm{0, 4, 15},
    XYPolyTerm{0, 6, -42},
    XYPolyTerm{0, 8, 28}};
};
template <>
struct Zernike<6, 8> {
  typedef std::integral_constant<int, 6> M;
  typedef std::integral_constant<unsigned, 8> N;
  static const constexpr std::array<XYPolyTerm, 8> terms{
    XYPolyTerm{8, 0, -8},
    XYPolyTerm{6, 0, 7},
    XYPolyTerm{6, 2, 112},
    XYPolyTerm{4, 2, -105},
    XYPolyTerm{2, 4, 105},
    XYPolyTerm{2, 6, -112},
    XYPolyTerm{0, 6, -7},
    XYPolyTerm{0, 8, 8}};
};
template <>
struct Zernike<8, 8> {
  typedef std::integral_constant<int, 8> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{8, 0, 1},
    XYPolyTerm{6, 2, -28},
    XYPolyTerm{4, 4, 70},
    XYPolyTerm{2, 6, -28},
    XYPolyTerm{0, 8, 1}};
};
template <>
struct Zernike<-9, 9> {
  typedef std::integral_constant<int, -9> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{9, 0, 1},
    XYPolyTerm{7, 2, -36},
    XYPolyTerm{5, 4, 126},
    XYPolyTerm{3, 6, -84},
    XYPolyTerm{1, 8, 9}};
};
template <>
struct Zernike<-7, 9> {
  typedef std::integral_constant<int, -7> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 9> terms{
    XYPolyTerm{9, 0, -9},
    XYPolyTerm{7, 0, 8},
    XYPolyTerm{7, 2, 180},
    XYPolyTerm{5, 2, -168},
    XYPolyTerm{5, 4, -126},
    XYPolyTerm{3, 4, 280},
    XYPolyTerm{3, 6, -252},
    XYPolyTerm{1, 6, -56},
    XYPolyTerm{1, 8, 63}};
};
template <>
struct Zernike<-5, 9> {
  typedef std::integral_constant<int, -5> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 11> terms{
    XYPolyTerm{9, 0, 36},
    XYPolyTerm{7, 0, -56},
    XYPolyTerm{7, 2, -288},
    XYPolyTerm{5, 0, 21},
    XYPolyTerm{5, 2, 504},
    XYPolyTerm{5, 4, -504},
    XYPolyTerm{3, 2, -210},
    XYPolyTerm{3, 4, 280},
    XYPolyTerm{1, 4, 105},
    XYPolyTerm{1, 6, -280},
    XYPolyTerm{1, 8, 180}};
};
template <>
struct Zernike<-3, 9> {
  typedef std::integral_constant<int, -3> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 13> terms{
    XYPolyTerm{9, 0, -84},
    XYPolyTerm{7, 0, 168},
    XYPolyTerm{5, 0, -105},
    XYPolyTerm{5, 2, -168},
    XYPolyTerm{5, 4, 504},
    XYPolyTerm{3, 0, 20},
    XYPolyTerm{3, 2, 210},
    XYPolyTerm{3, 4, -840},
    XYPolyTerm{3, 6, 672},
    XYPolyTerm{1, 2, -60},
    XYPolyTerm{1, 4, 315},
    XYPolyTerm{1, 6, -504},
    XYPolyTerm{1, 8, 252}};
};
template <>
struct Zernike<-1, 9> {
  typedef std::integral_constant<int, -1> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 15> terms{
    XYPolyTerm{9, 0, 126},
    XYPolyTerm{7, 0, -280},
    XYPolyTerm{7, 2, 504},
    XYPolyTerm{5, 0, 210},
    XYPolyTerm{5, 2, -840},
    XYPolyTerm{5, 4, 756},
    XYPolyTerm{3, 0, -60},
    XYPolyTerm{3, 2, 420},
    XYPolyTerm{3, 4, -840},
    XYPolyTerm{3, 6, 504},
    XYPolyTerm{1, 0, 5},
    XYPolyTerm{1, 2, -60},
    XYPolyTerm{1, 4, 210},
    XYPolyTerm{1, 6, -280},
    XYPolyTerm{1, 8, 126}};
};
template <>
struct Zernike<1,9> {
  typedef std::integral_constant<int, 1> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 15> terms{
    XYPolyTerm{8, 1, 126},
    XYPolyTerm{6, 1, -280},
    XYPolyTerm{6, 3, 504},
    XYPolyTerm{4, 1, 210},
    XYPolyTerm{4, 3, -840},
    XYPolyTerm{4, 5, 756},
    XYPolyTerm{2, 1, -60},
    XYPolyTerm{2, 3, 420},
    XYPolyTerm{2, 5, -840},
    XYPolyTerm{2, 7, 504},
    XYPolyTerm{0, 1, 5},
    XYPolyTerm{0, 3, -60},
    XYPolyTerm{0, 5, 210},
    XYPolyTerm{0, 7, -280},
    XYPolyTerm{0, 9, 126}};
};
template <>
struct Zernike<3, 9> {
  typedef std::integral_constant<int, 3> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 13> terms{
    XYPolyTerm{8, 1, -252},
    XYPolyTerm{6, 1, 504},
    XYPolyTerm{6, 3, -672},
    XYPolyTerm{4, 1, -315},
    XYPolyTerm{4, 3, 840},
    XYPolyTerm{4, 5, -504},
    XYPolyTerm{2, 1, 60},
    XYPolyTerm{2, 3, -210},
    XYPolyTerm{2, 5, 168},
    XYPolyTerm{0, 3, -20},
    XYPolyTerm{0, 5, 105},
    XYPolyTerm{0, 7, -168},
    XYPolyTerm{0, 9, 84}};
};
template <>
struct Zernike<5, 9> {
  typedef std::integral_constant<int, 5> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 11> terms{
    XYPolyTerm{8, 1, 180},
    XYPolyTerm{6, 1, -280},
    XYPolyTerm{4, 1, 105},
    XYPolyTerm{4, 3, 280},
    XYPolyTerm{4, 5, -504},
    XYPolyTerm{2, 3, -210},
    XYPolyTerm{2, 5, 504},
    XYPolyTerm{2, 7, -288},
    XYPolyTerm{0, 5, 21},
    XYPolyTerm{0, 7, -56},
    XYPolyTerm{0, 9, 36}};
};
template <>
struct Zernike<7, 9> {
  typedef std::integral_constant<int, 7> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 9> terms{
    XYPolyTerm{8, 1, -63},
    XYPolyTerm{6, 1, 56},
    XYPolyTerm{6, 3, -252},
    XYPolyTerm{4, 3, -280},
    XYPolyTerm{4, 5, 126},
    XYPolyTerm{2, 5, 168},
    XYPolyTerm{2, 7, -180},
    XYPolyTerm{0, 7, -8},
    XYPolyTerm{0, 9, 9}};
};
template <>
struct Zernike<9, 9> {
  typedef std::integral_constant<int, 9> M;
  typedef std::integral_constant<unsigned, 9> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{8, 1, 9},
    XYPolyTerm{6, 3, -84},
    XYPolyTerm{4, 5, 126},
    XYPolyTerm{2, 7, -36},
    XYPolyTerm{0, 9, 1}};
};
template <>
struct Zernike<-10, 10> {
  typedef std::integral_constant<int, -10> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 5> terms{
    XYPolyTerm{9, 1, 10},
    XYPolyTerm{7, 3, -120},
    XYPolyTerm{5, 5, 252},
    XYPolyTerm{3, 7, -120},
    XYPolyTerm{1, 9, 10}};
};
template <>
struct Zernike<-8, 10> {
  typedef std::integral_constant<int, -8> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 8> terms{
    XYPolyTerm{9, 1, -80},
    XYPolyTerm{7, 1, 72},
    XYPolyTerm{7, 3, 480},
    XYPolyTerm{5, 3, -504},
    XYPolyTerm{3, 5, 504},
    XYPolyTerm{3, 7, -480},
    XYPolyTerm{1, 7, -72},
    XYPolyTerm{1, 9, 80}};
};
template <>
struct Zernike<-6, 10> {
  typedef std::integral_constant<int, -6> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 12> terms{
    XYPolyTerm{9, 1, 270},
    XYPolyTerm{7, 1, -432},
    XYPolyTerm{7, 3, -360},
    XYPolyTerm{5, 1, 168},
    XYPolyTerm{5, 3, 1008},
    XYPolyTerm{5, 5, -1260},
    XYPolyTerm{3, 3, -560},
    XYPolyTerm{3, 5, 1008},
    XYPolyTerm{3, 7, -360},
    XYPolyTerm{1, 5, 168},
    XYPolyTerm{1, 7, -432},
    XYPolyTerm{1, 9, 270}};
};
template <>
struct Zernike<-4, 10> {
  typedef std::integral_constant<int, -4> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 12> terms{
    XYPolyTerm{9, 1, -480},
    XYPolyTerm{7, 1, 1008},
    XYPolyTerm{7, 3, -960},
    XYPolyTerm{5, 1, -672},
    XYPolyTerm{5, 3, 1008},
    XYPolyTerm{3, 1, 140},
    XYPolyTerm{3, 5, -1008},
    XYPolyTerm{3, 7, 960},
    XYPolyTerm{1, 3, -140},
    XYPolyTerm{1, 5, 672},
    XYPolyTerm{1, 7, -1008},
    XYPolyTerm{1, 9, 480}};
};
template <>
struct Zernike<-2, 10> {
  typedef std::integral_constant<int, -2> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 15> terms{
    XYPolyTerm{9, 1, 420},
    XYPolyTerm{7, 1, -1008},
    XYPolyTerm{7, 3, 1680},
    XYPolyTerm{5, 1, 840},
    XYPolyTerm{5, 3, -3024},
    XYPolyTerm{5, 5, 2520},
    XYPolyTerm{3, 1, -280},
    XYPolyTerm{3, 3, 1680},
    XYPolyTerm{3, 5, -3024},
    XYPolyTerm{3, 7, 1680},
    XYPolyTerm{1, 1, 30},
    XYPolyTerm{1, 3, -280},
    XYPolyTerm{1, 5, 840},
    XYPolyTerm{1, 7, -1008},
    XYPolyTerm{1, 9, 420}};
};
template <>
struct Zernike<0, 10> {
  typedef std::integral_constant<int, 0> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 21> terms{
    XYPolyTerm{10, 0, 252},
    XYPolyTerm{8, 0, -630},
    XYPolyTerm{8, 2, 1260},
    XYPolyTerm{6, 0, 560},
    XYPolyTerm{6, 2, -2520},
    XYPolyTerm{6, 4, 2520},
    XYPolyTerm{4, 0, -210},
    XYPolyTerm{4, 2, 1680},
    XYPolyTerm{4, 4, -3780},
    XYPolyTerm{4, 6, 2520},
    XYPolyTerm{2, 0, 30},
    XYPolyTerm{2, 2, -420},
    XYPolyTerm{2, 4, 1680},
    XYPolyTerm{2, 6, -2520},
    XYPolyTerm{2, 8, 1260},
    XYPolyTerm{0, 0, -1},
    XYPolyTerm{0, 2, 30},
    XYPolyTerm{0, 4, -210},
    XYPolyTerm{0, 6, 560},
    XYPolyTerm{0, 8, -630},
    XYPolyTerm{0, 10, 252}};
};
template <>
struct Zernike<2, 10> {
  typedef std::integral_constant<int, 2> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 18> terms{
    XYPolyTerm{10, 0, -210},
    XYPolyTerm{8, 0, 504},
    XYPolyTerm{8, 2, -630},
    XYPolyTerm{6, 0, -420},
    XYPolyTerm{6, 2, 1008},
    XYPolyTerm{6, 4, -420},
    XYPolyTerm{4, 0, 140},
    XYPolyTerm{4, 2, -420},
    XYPolyTerm{4, 6, 420},
    XYPolyTerm{2, 0, -15},
    XYPolyTerm{2, 4, 420},
    XYPolyTerm{2, 6, -1008},
    XYPolyTerm{2, 8, 630},
    XYPolyTerm{0, 2, 15},
    XYPolyTerm{0, 4, -140},
    XYPolyTerm{0, 6, 420},
    XYPolyTerm{0, 8, -504},
    XYPolyTerm{0, 10, 210}};
};
template <>
struct Zernike<4, 10> {
  typedef std::integral_constant<int, 4> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 18> terms{
    XYPolyTerm{10, 0, 120},
    XYPolyTerm{8, 0, -252},
    XYPolyTerm{8, 2, -360},
    XYPolyTerm{6, 0, 168},
    XYPolyTerm{6, 2, 1008},
    XYPolyTerm{6, 4, -1680},
    XYPolyTerm{4, 0, -35},
    XYPolyTerm{4, 2, -840},
    XYPolyTerm{4, 4, 2520},
    XYPolyTerm{4, 6, -1680},
    XYPolyTerm{2, 2, 210},
    XYPolyTerm{2, 4, -840},
    XYPolyTerm{2, 6, 1008},
    XYPolyTerm{2, 8, -360},
    XYPolyTerm{0, 4, -35},
    XYPolyTerm{0, 6, 168},
    XYPolyTerm{0, 8, -252},
    XYPolyTerm{0, 10, 120}};
};
template <>
struct Zernike<6, 10> {
  typedef std::integral_constant<int, 6> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 14> terms{
    XYPolyTerm{10, 0, -45},
    XYPolyTerm{8, 0, 72},
    XYPolyTerm{8, 2, 585},
    XYPolyTerm{6, 0, -28},
    XYPolyTerm{6, 2, -1008},
    XYPolyTerm{6, 4, 630},
    XYPolyTerm{4, 2, 420},
    XYPolyTerm{4, 6, -630},
    XYPolyTerm{2, 4, -420},
    XYPolyTerm{2, 6, 1008},
    XYPolyTerm{2, 8, -585},
    XYPolyTerm{0, 6, 28},
    XYPolyTerm{0, 8, -72},
    XYPolyTerm{0, 10, 45}};
};
template <>
struct Zernike<8, 10> {
  typedef std::integral_constant<int, 8> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 11> terms{
    XYPolyTerm{10, 0, 10},
    XYPolyTerm{8, 0, -9},
    XYPolyTerm{8, 2, -270},
    XYPolyTerm{6, 2, 252},
    XYPolyTerm{6, 4, 420},
    XYPolyTerm{4, 4, -630},
    XYPolyTerm{4, 6, 420},
    XYPolyTerm{2, 6, 252},
    XYPolyTerm{2, 8, -270},
    XYPolyTerm{0, 8, -9},
    XYPolyTerm{0, 10, 10}};
};
template <>
struct Zernike<10, 10> {
  typedef std::integral_constant<int, 10> M;
  typedef std::integral_constant<unsigned, 10> N;
  static const constexpr std::array<XYPolyTerm, 6> terms{
    XYPolyTerm{10, 0, -1},
    XYPolyTerm{8, 2, 45},
    XYPolyTerm{6, 4, -210},
    XYPolyTerm{4, 6, 210},
    XYPolyTerm{2, 8, -45},
    XYPolyTerm{0, 10, 1}};
};

typedef std::integral_constant<unsigned, 10> zernike_max_order;

template <typename T, unsigned N, typename ZS>
struct zernike_series_ {
  //typedef ... value;
};

template <typename T, typename...Zs>
struct zernike_series {
  typedef zernike_series_<T, 0, zernike_series<T, Zs...>> series_;
  typedef typename series_::max_n max_n;
  constexpr static unsigned D = ((max_n::value + 1) * (max_n::value + 2)) / 2;
  static void
  expand(
    std::experimental::mdspan<T, max_n::value + 1, max_n::value + 1>& ary,
    const std::experimental::mdspan<T, D>& coefficients) {
    for (size_t i = 0; i <= max_n::value; ++i)
      for (size_t j = 0; j <= max_n::value; ++j)
        ary(i, j) = (T)0;
    zernike_series_<T, max_n::value, zernike_series<T, Zs...>>::
      expand(ary, coefficients);
  }
#ifdef HYPERION_USE_KOKKOS
  template <typename ...Args>
  static void
  expand(
    const Kokkos::View<T**, Args...>& ary,
    const Kokkos::View<const T*, Args...>& coefficients) {
    assert(ary.extent(0) == max_n::value + 1);
    assert(ary.extent(1) == max_n::value + 1);
    assert(coefficients.extent(0) == D);
    zernike_series_<T, max_n::value, zernike_series<T, Zs...>>::
      expand(ary, coefficients);
  }
#endif // HYPERION_USE_KOKKOS
};

template <typename T, unsigned N>
struct zernike_series_<T, N, zernike_series<T>> {
  typedef typename std::integral_constant<unsigned, N> max_n;
  constexpr static unsigned D = ((N + 1) * (N + 2)) / 2;
  static void
  expand(
    std::experimental::mdspan<T, N+1, N+1>&,
    const std::experimental::mdspan<T, D>&) {}
#ifdef HYPERION_USE_KOKKOS
  template <typename ...Args>
  static void
  expand(
    const Kokkos::View<T**, Args...>&,
    const Kokkos::View<const T*, Args...>&) {}
#endif // HYPERION_USE_KOKKOS
};

template <typename T, unsigned N0, typename Z, typename...Zs>
struct zernike_series_<T, N0, zernike_series<T, Z, Zs...>> {
  typedef typename std::conditional<
    (N0 >= Z::N::value),
    typename zernike_series_<T, N0, zernike_series<T, Zs...>>::max_n,
    typename zernike_series_<T, Z::N::value, zernike_series<T, Zs...>>::max_n>
  ::type max_n;
  constexpr static unsigned D = ((N0 + 1) * (N0 + 2)) / 2;

  static void
  expand(
    std::experimental::mdspan<T, N0+1, N0+1>& ary,
    const std::experimental::mdspan<T, D>& coeffs) {
    const auto& c = coeffs(D - (sizeof...(Zs) + 1));
    for (auto& t : Z::terms)
      ary(t.px, t.py) += c * t.c;
    zernike_series_<T, N0, zernike_series<T, Zs...>>::expand(ary, coeffs);
  }
#ifdef HYPERION_USE_KOKKOS
  template <typename ...Args>
  static void
  expand(
    const Kokkos::View<T**, Args...>& ary,
    const Kokkos::View<const T*, Args...>& coeffs) {
    assert(ary.extent(0) == N0 + 1);
    assert(ary.extent(1) == N0 + 1);
    assert(coeffs.extent(0) == D);
    const auto& c = coeffs(D - (sizeof...(Zs) + 1));
    for (auto& t : Z::terms)
      ary(t.px, t.py) += c * t.c;
    zernike_series_<T, N0, zernike_series<T, Zs...>>::expand(ary, coeffs);
  }
#endif // HYPERION_USE_KOKKOS
};

template <typename T, typename ZP, typename...Zs>
struct zernike_basis_ {
};
template <typename T, typename...PZs, typename...Zs>
struct zernike_basis_<T, zernike_series<T, PZs...>, Zs...> {
  typedef zernike_series<T, PZs..., Zs...> series;
};

template <typename T, unsigned N, typename Z>
struct zernike_basis_base {
  constexpr static unsigned num_terms = (N+1)*(N+2)/2;
  constexpr static unsigned degree = N;
  typedef T var_t;
  static void
  expand(
    std::experimental::mdspan<T, N+1, N+1>& ary,
    const std::experimental::mdspan<T, num_terms>& coefficients) {
    Z::series::expand(ary, coefficients);
  }
  static void
  expand(
    std::experimental::mdspan<T, N+1, N+1>&& ary,
    std::experimental::mdspan<T, num_terms>&& coefficients) {
    Z::series::expand(ary, coefficients);
  }
#ifdef HYPERION_USE_KOKKOS
  template <typename ...Args>
  static void
  expand(
    const Kokkos::View<T**, Args...>& ary,
    const Kokkos::View<const T*, Args...>& coefficients) {
    assert(ary.extent(0) == N + 1);
    assert(ary.extent(1) == N + 1);
    assert(coefficients.extent(0) == num_terms);
    Z::series::expand(ary, coefficients);
  }
#endif // HYPERION_USE_KOKKOS
  constexpr static unsigned
  index(int m, unsigned n) {
    return (n * (n + 2) + m) / 2;
  }
};

template <typename T, unsigned N>
struct zernike_basis {
  typedef void series;
};
template <typename T>
struct zernike_basis<T, 0>
  : public zernike_basis_base<T, 0, zernike_basis<T, 0>> {
  typedef zernike_series<T, Zernike<0,0>> series;
};
template <typename T>
struct zernike_basis<T, 1>
  : public zernike_basis_base<T, 1, zernike_basis<T, 1>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 0>::series,
    Zernike<-1,1>,Zernike<1,1>>::series series;
};
template <typename T>
struct zernike_basis<T, 2>
  : public zernike_basis_base<T, 2, zernike_basis<T, 2>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 1>::series,
    Zernike<-2,2>,Zernike<0,2>,Zernike<2,2>>::series series;
};
template <typename T>
struct zernike_basis<T, 3>
  : public zernike_basis_base<T, 3, zernike_basis<T, 3>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 2>::series,
    Zernike<-3,3>,Zernike<-1,3>,Zernike<1,3>, Zernike<3,3>>::series series;
};
template <typename T>
struct zernike_basis<T, 4>
  : public zernike_basis_base<T, 4, zernike_basis<T, 4>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 3>::series,
    Zernike<-4,4>,Zernike<-2,4>,Zernike<0,4>,
    Zernike<2,4>,Zernike<4,4>>::series series;
};
template <typename T>
struct zernike_basis<T, 5>
  : public zernike_basis_base<T, 5, zernike_basis<T, 5>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 4>::series,
    Zernike<-5,5>,Zernike<-3,5>,Zernike<-1,5>,
    Zernike<1,5>,Zernike<3,5>,Zernike<5,5>>::series series;
};
template <typename T>
struct zernike_basis<T, 6>
  : public zernike_basis_base<T, 6, zernike_basis<T, 6>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 5>::series,
    Zernike<-6,6>,Zernike<-4,6>,Zernike<-2,6>,Zernike<0,6>,
    Zernike<2,6>,Zernike<4,6>,Zernike<6,6>>::series series;
};
template <typename T>
struct zernike_basis<T, 7>
  : public zernike_basis_base<T, 7, zernike_basis<T, 7>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 6>::series,
    Zernike<-7,7>,Zernike<-5,7>,Zernike<-3,7>,Zernike<-1,7>,
    Zernike<1,7>,Zernike<3,7>,Zernike<5,7>,Zernike<7,7>>::series series;
};
template <typename T>
struct zernike_basis<T, 8>
  : public zernike_basis_base<T, 8, zernike_basis<T, 8>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 7>::series,
    Zernike<-8,8>,Zernike<-6,8>,Zernike<-4,8>,Zernike<-2,8>,Zernike<0,8>,
    Zernike<2,8>,Zernike<4,8>,Zernike<6,8>,Zernike<8,8>>::series series;
};
template <typename T>
struct zernike_basis<T, 9>
  : public zernike_basis_base<T, 9, zernike_basis<T, 9>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 8>::series,
    Zernike<-9,9>,Zernike<-7,9>,Zernike<-5,9>,Zernike<-3,9>,Zernike<-1,9>,
    Zernike<1,9>,Zernike<3,9>,Zernike<5,9>,Zernike<7,9>,
    Zernike<9,9>>::series series;
};
template <typename T>
struct zernike_basis<T, 10>
  : public zernike_basis_base<T, 10, zernike_basis<T, 10>> {
  typedef typename zernike_basis_<
    T,
    typename zernike_basis<T, 9>::series,
    Zernike<-10,10>,Zernike<-8,10>,Zernike<-6,10>,Zernike<-4,10>,Zernike<-2,10>,
    Zernike<0,10>,Zernike<2,10>,Zernike<4,10>,Zernike<6,10>,Zernike<8,10>,
    Zernike<10,10>>::series series;
};

constexpr inline unsigned
zernike_num_terms(unsigned order) {
  switch (order) {
#define NT(N) \
  case N: \
    return zernike_basis<float, N>::num_terms;   \
    break
  NT(0);
  NT(1);
  NT(2);
  NT(3);
  NT(4);
  NT(5);
  NT(6);
  NT(7);
  NT(8);
  NT(9);
  NT(10);
  default:
    assert(false);
    return 0;
    break;
#undef NT
  }
}

constexpr inline unsigned
zernike_index(int m, unsigned n) {
  return (n * (n + 2) + m) / 2;
}

constexpr std::pair<int, unsigned>
zernike_inverse_index(int i) {
  return
    ((static_cast<unsigned>(i)
      > zernike_index(9, 9))
     ? std::pair(2 * i - 10 * 12, 10)
     : ((static_cast<unsigned>(i)
         > zernike_index(8, 8)
         ? std::pair(2 * i - 9 * 11, 9)
         : ((static_cast<unsigned>(i)
             > zernike_index(7, 7)
             ? std::pair(2 * i - 8 * 10, 8)
             : ((static_cast<unsigned>(i)
                 > zernike_index(6, 6)
                 ? std::pair(2 * i - 7 * 9, 7)
                 : ((static_cast<unsigned>(i)
                     > zernike_index(5, 5)
                     ? std::pair(2 * i - 6 * 8, 6)
                     : ((static_cast<unsigned>(i)
                         > zernike_index(4, 4)
                         ? std::pair(2 * i - 5 * 7, 5)
                         : ((static_cast<unsigned>(i)
                             > zernike_index(3, 3)
                             ? std::pair(2 * i - 4 * 6, 4)
                             : ((static_cast<unsigned>(i)
                                 > zernike_index(2, 2)
                                 ? std::pair(2 * i - 3 * 5, 3)
                                 : ((static_cast<unsigned>(i)
                                     > zernike_index(1, 1)
                                     ? std::pair(2 * i - 2 * 4, 2)
                                     : ((static_cast<unsigned>(i)
                                         > zernike_index(0, 0))
                                        ? std::pair(2 * i - 1 * 3, 1)
                                        : std::pair(0, 0)))))))))))))))))));
}

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_ZERNIKE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
