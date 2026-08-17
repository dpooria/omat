#include <fftw3.h>

#define N 256

int main(void) {
    fftw_complex in[N], out[N];
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    return 0;
}
