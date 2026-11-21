#include <complex>
#include <cstdint>

#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft/pocketfft_hdronly.h"

struct Plan {
  pocketfft::shape_t shape, axes;
  pocketfft::stride_t stride_in, stride_out;
  int kind;
  bool single, inverse;
  size_t threads;
};

extern "C" void *omat_pocketfft_plan_create(intptr_t n, const intptr_t *shape,
                                            const intptr_t *stride_in,
                                            const intptr_t *stride_out,
                                            intptr_t first, int kind,
                                            bool single, bool inverse,
                                            intptr_t threads) {
  auto *p = new Plan;
  p->shape.assign(shape, shape + n);
  p->stride_in.assign(stride_in, stride_in + n);
  p->stride_out.assign(stride_out, stride_out + n);
  for (intptr_t i = first; i < n; ++i)
    p->axes.push_back(i);
  p->kind = kind;
  p->single = single;
  p->inverse = inverse;
  p->threads = threads;
  return p;
}

template <typename T>
void execute(const Plan &p, const void *input, void *output) {
  using C = std::complex<T>;
  switch (p.kind) {
  case 0:
    pocketfft::c2c(p.shape, p.stride_in, p.stride_out, p.axes, !p.inverse,
                   static_cast<const C *>(input), static_cast<C *>(output),
                   T(1), p.threads);
    break;
  case 1:
    pocketfft::r2c(p.shape, p.stride_in, p.stride_out, p.axes, true,
                   static_cast<const T *>(input), static_cast<C *>(output),
                   T(1), p.threads);
    break;
  case 2:
    pocketfft::c2r(p.shape, p.stride_in, p.stride_out, p.axes, false,
                   static_cast<const C *>(input), static_cast<T *>(output),
                   T(1), p.threads);
    break;
  }
}

extern "C" void omat_pocketfft_execute(const Plan *p, const void *input,
                                       void *output) {
  if (p->single)
    execute<float>(*p, input, output);
  else
    execute<double>(*p, input, output);
}

extern "C" void omat_pocketfft_destroy(Plan *p) { delete p; }
