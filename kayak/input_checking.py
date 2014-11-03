from warnings import warn

def check_equal_ndims_for_broadcasting(obj):
    ndims = [p.value.ndim for p in obj._parents]
    if not all([ndims[0] == ndims_other for ndims_other in ndims[1:]]):
        p_shapes = [p.shape for p in obj._parents]
        warn(("Broadcasting arrays with shapes %s " +
              "by prepending singleton dimensions.") % p_shapes,
             stacklevel=2)
