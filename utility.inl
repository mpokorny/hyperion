template <int IPDIM, int PRJDIM>
Legion::IndexPartitionT<PRJDIM>
ip_down(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<PRJDIM> ispace,
  const std::array<int, PRJDIM>& dmap) {

  // dmap is a map from a dimension (index) in ispace to a dimension (index) in
  // the parent index space of ipart

  // TODO: how to ensure that the shape of ispace is commensurate with the
  // subspace of ipart's parent index space chosen by dmap? This implementation
  // should be safe when the subspace requirement is not met, but the result may
  // not be what is expected in such cases.

  static_assert(PRJDIM <= IPDIM);

  Legion::FieldSpace fs = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(Legion::Rect<IPDIM>), 0);
  }
  Legion::LogicalRegionT<PRJDIM> lr =
    runtime->create_logical_region(ctx, ispace, fs);

  {
    Legion::IndexSpaceT<IPDIM> ipart_is(
      runtime->get_parent_index_space(ctx, ipart));
    Legion::Domain ipart_domain =
      runtime->get_index_space_domain(ctx, ipart_is);
    assert(ipart_domain.get_dim() == IPDIM);
    Legion::Rect<IPDIM> ipart_bounds =
      ipart_domain.bounds<IPDIM, Legion::coord_t>();
    auto filler =
      Legion::InlineLauncher(
        Legion::RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    filler.add_field(0);
    auto pr = runtime->map_region(ctx, filler);
    Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Rect<IPDIM>,
      PRJDIM,
      Legion::coord_t,
      Legion::AffineAccessor<Legion::Rect<IPDIM>, PRJDIM, Legion::coord_t>,
      false> values(pr, 0);
    Legion::DomainT<PRJDIM> d = runtime->get_index_space_domain(ctx, ispace);
    for (Legion::PointInDomainIterator<PRJDIM> pid(d);
         pid();
         pid++) {

      Legion::Rect<IPDIM> r = ipart_bounds;
      for (size_t i = 0; i < PRJDIM; ++i) {
        r.lo[dmap[i]] = pid[i];
        r.hi[dmap[i]] = pid[i];
      }
      values[*pid] = r;
    }
    runtime->unmap_region(ctx, pr);
  }
  auto result =
    runtime->create_partition_by_preimage_range(
      ctx,
      ipart,
      lr,
      lr,
      0,
      runtime->get_index_partition_color_space_name(ctx, ipart));
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  return Legion::IndexPartitionT<PRJDIM>(result);
}

template <int IPDIM, int PRJDIM>
Legion::IndexPartitionT<PRJDIM>
ip_up(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const Legion::IndexPartition ipart,
  const Legion::IndexSpaceT<PRJDIM> ispace,
  const std::array<int, PRJDIM>& dmap) {

  // dmap is a map from a dimension (index) in ispace to a dimension (index) in
  // the parent index space of ipart, using values of -1 to indicate free or
  // unmapped dimensions; this array must contain all the values from -1 to
  // IPDIM - 1 (inclusive)

  // TODO: how to ensure that the shape of the subspace of ipart's parent index
  // space selected by dmap is commensurate with ispace? This implementation
  // should be safe when the subspace requirement is not met, but the result may
  // not be what is expected in such cases.

  static_assert(IPDIM < PRJDIM);

  Legion::FieldSpace fs = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(Legion::Rect<PRJDIM>), 0);
  }
  Legion::IndexSpaceT<IPDIM> ipart_is(
    runtime->get_parent_index_space(ctx, ipart));
  Legion::LogicalRegionT<IPDIM> lr =
    runtime->create_logical_region(ctx, ipart_is, fs);

  {
    Legion::Domain ispace_domain =
      runtime->get_index_space_domain(ctx, ispace);
    assert(ispace_domain.get_dim() == PRJDIM);
    Legion::Rect<PRJDIM> ispace_bounds =
      ispace_domain.bounds<PRJDIM, Legion::coord_t>();
    auto filler =
      Legion::InlineLauncher(
        Legion::RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    filler.add_field(0);
    auto pr = runtime->map_region(ctx, filler);
    Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Rect<PRJDIM>,
      IPDIM,
      Legion::coord_t,
      Legion::AffineAccessor<Legion::Rect<PRJDIM>, IPDIM, Legion::coord_t>,
      false> values(pr, 0);
    Legion::DomainT<IPDIM> d = runtime->get_index_space_domain(ctx, ipart_is);
    for (Legion::PointInDomainIterator<IPDIM> pid(d);
         pid();
         pid++) {

      Legion::Rect<PRJDIM> r = ispace_bounds;
      for (size_t i = 0; i < PRJDIM; ++i) {
        if (dmap[i] != -1) {
          r.lo[i] = pid[dmap[i]];
          r.hi[i] = pid[dmap[i]];
        }
        values[*pid] = r;
      }
    }
    runtime->unmap_region(ctx, pr);
  }
  auto lp = runtime->get_logical_partition(ctx, lr, ipart);
  auto result =
    runtime->create_partition_by_image_range(
      ctx,
      ispace,
      lp,
      lr,
      0,
      runtime->get_index_partition_color_space_name(ctx, ipart));
  runtime->destroy_logical_partition(ctx, lp);
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  return Legion::IndexPartitionT<PRJDIM>(result);
}

template <> inline
Legion::IndexPartitionT<1>
projected_index_partition<1, 1>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<1> ispace,
  const std::array<int, 1>& dmap) {

  return ip_down<1, 1>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<1>
projected_index_partition<2, 1>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<1> ispace,
  const std::array<int, 1>& dmap) {

  return ip_down<2, 1>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<1>
projected_index_partition<3, 1>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<1> ispace,
  const std::array<int, 1>& dmap) {

  return ip_down<3, 1>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<2>
projected_index_partition<1, 2>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<2> ispace,
  const std::array<int, 2>& dmap) {

  return ip_up<1, 2>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<2>
projected_index_partition<2, 2>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<2> ispace,
  const std::array<int, 2>& dmap) {

  return ip_down<2, 2>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<2>
projected_index_partition<3, 2>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<2> ispace,
  const std::array<int, 2>& dmap) {

  return ip_down<3, 2>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<3>
projected_index_partition<1, 3>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<3> ispace,
  const std::array<int, 3>& dmap) {

  return ip_up<1, 3>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<3>
projected_index_partition<2, 3>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<3> ispace,
  const std::array<int, 3>& dmap) {

  return ip_up<2, 3>(ctx, runtime, ipart, ispace, dmap);
}

template <> inline
Legion::IndexPartitionT<3>
projected_index_partition<3, 3>(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ipart,
  Legion::IndexSpaceT<3> ispace,
  const std::array<int, 3>& dmap) {

  return ip_down<3, 3>(ctx, runtime, ipart, ispace, dmap);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
