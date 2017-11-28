#ifndef __NANMATH_H__
#define __NANMATH_H__

#ifdef __cplusplus
extern "C" {
#endif

void nan_corr_float(float *x, float *y, unsigned N, float *corr, unsigned *count);

void nan_corr_float_avx(float *x, float *y, unsigned N, float *corr, unsigned *count);

void nan_corr_double(double *x, double *y, unsigned N, double *corr, unsigned *count);

void nan_corr_double_avx(double *x, double *y, unsigned N, double *corr, unsigned *count);

void nan_mean_std_float(float *x, unsigned N, float *mean, float *std, unsigned *count);

void nan_mean_std_float_avx(float *x, unsigned N, float *mean, float *std, unsigned *count);

void nan_mean_std_double(double *x, unsigned N, double *mean, double *std, unsigned *count);

void nan_mean_std_double_avx(double *x, unsigned N, double *mean, double *std, unsigned *count);

#ifdef __cplusplus
}
#endif

#endif // __NANMATH_H__
