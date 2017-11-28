#include "nanmath.h"
#include <math.h>
#include <stdbool.h>
#include <immintrin.h>  // AVX

#define EPS 1e-12

inline bool fast_isnan_float(float v) {
    return (v != v);
}

inline bool fast_isnan_double(double v) {
    return (v != v);
}

void nan_corr_float(float *x, float *y, unsigned N, float *corr, unsigned *count) {
    float sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_x = 0.0, sum_y = 0.0;
    unsigned cnt = 0;
    for(unsigned i = 0; i < N; ++i) {
        float a = x[i], b = y[i];
        if(!fast_isnan_float(a) && !fast_isnan_float(b)) {
            sum_xy += a * b;
            sum_xx += a * a;
            sum_yy += b * b;
            sum_x += a;
            sum_y += b;
            ++cnt;
        }
    }

    *count = cnt;
    if(cnt > 1) {
        float cov_xy = sum_xy / cnt - (sum_x / cnt) * (sum_y / cnt);
        float var_x = sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt);
        float var_y = sum_yy / cnt - (sum_y / cnt) * (sum_y / cnt);
        if(var_x > EPS && var_y > EPS) {
            *corr = cov_xy / sqrt(var_x) / sqrt(var_y);
        } else {
            *corr = NAN;
        }
    } else {
        *corr = NAN;
    }
}

void nan_corr_float_avx(float *x, float *y, unsigned N, float *corr, unsigned *count) {
    unsigned BOUNDARY = (N / 8) * 8;
    unsigned index = 0;
    float sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_x = 0.0, sum_y = 0.0;
    unsigned cnt = 0;
    __m256 sum_xy_reg = _mm256_set1_ps(0.0);
    __m256 sum_xx_reg = _mm256_set1_ps(0.0);
    __m256 sum_yy_reg = _mm256_set1_ps(0.0);
    __m256 sum_x_reg = _mm256_set1_ps(0.0);
    __m256 sum_y_reg = _mm256_set1_ps(0.0);
    __m256 cnt_reg = _mm256_set1_ps(0.0);
    __m256 all_ones = _mm256_set1_ps(1.0);
    while(index < BOUNDARY) {
        __m256 a = _mm256_loadu_ps(x + index);
        __m256 b = _mm256_loadu_ps(y + index);
        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(a, a, _CMP_EQ_OQ), _mm256_cmp_ps(b, b, _CMP_EQ_OQ));
        a = _mm256_and_ps(mask, a);
        b = _mm256_and_ps(mask, b);
        sum_xy_reg = _mm256_fmadd_ps(a, b, sum_xy_reg);
        sum_xx_reg = _mm256_fmadd_ps(a, a, sum_xx_reg);
        sum_yy_reg = _mm256_fmadd_ps(b, b, sum_yy_reg);
        sum_x_reg = _mm256_add_ps(a, sum_x_reg);
        sum_y_reg = _mm256_add_ps(b, sum_y_reg);
        cnt_reg = _mm256_add_ps(_mm256_and_ps(mask, all_ones), cnt_reg);
        index += 8;
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, sum_xy_reg);
    sum_xy = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, sum_xx_reg);
    sum_xx = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, sum_yy_reg);
    sum_yy = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, sum_x_reg);
    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, sum_y_reg);
    sum_y = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, cnt_reg);
    cnt = (unsigned)(round(buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7]));

    while(index < N) {
        float a = x[index], b = y[index];
        if(!fast_isnan_float(a) && !fast_isnan_float(b)) {
            sum_xy += a * b;
            sum_xx += a * a;
            sum_yy += b * b;
            sum_x += a;
            sum_y += b;
            ++cnt;;
        }
        ++index;
    }

    *count = cnt;
    if(cnt > 1) {
        float cov_xy = sum_xy / cnt - (sum_x / cnt) * (sum_y / cnt);
        float var_x = sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt);
        float var_y = sum_yy / cnt - (sum_y / cnt) * (sum_y / cnt);
        if(var_x > EPS && var_y > EPS) {
            *corr = cov_xy / sqrt(var_x) / sqrt(var_y);
        } else {
            *corr = NAN;
        }
    } else {
        *corr = NAN;
    }
}

void nan_corr_double(double *x, double *y, unsigned N, double *corr, unsigned *count) {
    double sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_x = 0.0, sum_y = 0.0;
    unsigned cnt = 0;
    for(unsigned i = 0; i < N; ++i) {
        double a = x[i], b = y[i];
        if(!fast_isnan_double(a) && !fast_isnan_double(b)) {
            sum_xy += a * b;
            sum_xx += a * a;
            sum_yy += b * b;
            sum_x += a;
            sum_y += b;
            ++cnt;;
        }
    }

    *count = cnt;
    if(cnt > 1) {
        double cov_xy = sum_xy / cnt - (sum_x / cnt) * (sum_y / cnt);
        double var_x = sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt);
        double var_y = sum_yy / cnt - (sum_y / cnt) * (sum_y / cnt);
        if(var_x > EPS && var_y > EPS) {
            *corr = cov_xy / sqrt(var_x) / sqrt(var_y);
        } else {
            *corr = NAN;
        }
    } else {
        *corr = NAN;
    }
}

void nan_corr_double_avx(double *x, double *y, unsigned N, double *corr, unsigned *count) {
    unsigned BOUNDARY = (N / 4) * 4;
    unsigned index = 0;
    double sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_x = 0.0, sum_y = 0.0;
    unsigned cnt = 0;
    __m256d sum_xy_reg = _mm256_set1_pd(0.0);
    __m256d sum_xx_reg = _mm256_set1_pd(0.0);
    __m256d sum_yy_reg = _mm256_set1_pd(0.0);
    __m256d sum_x_reg = _mm256_set1_pd(0.0);
    __m256d sum_y_reg = _mm256_set1_pd(0.0);
    __m256d cnt_reg = _mm256_set1_pd(0.0);
    __m256d all_ones = _mm256_set1_pd(1.0);
    while(index < BOUNDARY) {
        __m256d a = _mm256_loadu_pd(x + index);
        __m256d b = _mm256_loadu_pd(y + index);
        __m256d mask = _mm256_and_pd(_mm256_cmp_pd(a, a, _CMP_EQ_OQ), _mm256_cmp_pd(b, b, _CMP_EQ_OQ));
        a = _mm256_and_pd(mask, a);
        b = _mm256_and_pd(mask, b);
        sum_xy_reg = _mm256_fmadd_pd(a, b, sum_xy_reg);
        sum_xx_reg = _mm256_fmadd_pd(a, a, sum_xx_reg);
        sum_yy_reg = _mm256_fmadd_pd(b, b, sum_yy_reg);
        sum_x_reg = _mm256_add_pd(a, sum_x_reg);
        sum_y_reg = _mm256_add_pd(b, sum_y_reg);
        cnt_reg = _mm256_add_pd(_mm256_and_pd(mask, all_ones), cnt_reg);
        index += 4;
    }
    double buffer[4];
    _mm256_storeu_pd(buffer, sum_xy_reg);
    sum_xy = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, sum_xx_reg);
    sum_xx = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, sum_yy_reg);
    sum_yy = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, sum_x_reg);
    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, sum_y_reg);
    sum_y = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, cnt_reg);
    cnt = (unsigned)(round(buffer[0] + buffer[1] + buffer[2] + buffer[3]));

    while(index < N) {
        double a = x[index], b = y[index];
        if(!fast_isnan_double(a) && !fast_isnan_double(b)) {
            sum_xy += a * b;
            sum_xx += a * a;
            sum_yy += b * b;
            sum_x += a;
            sum_y += b;
            ++cnt;;
        }
        ++index;
    }

    *count = cnt;
    if(cnt > 1) {
        double cov_xy = sum_xy / cnt - (sum_x / cnt) * (sum_y / cnt);
        double var_x = sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt);
        double var_y = sum_yy / cnt - (sum_y / cnt) * (sum_y / cnt);
        if(var_x > EPS && var_y > EPS) {
            *corr = cov_xy / sqrt(var_x) / sqrt(var_y);
        } else {
            *corr = NAN;
        }
    } else {
        *corr = NAN;
    }
}

void nan_mean_std_float(float *x, unsigned N, float *mean, float *std, unsigned *count) {
    float sum_xx = 0.0, sum_x = 0.0;
    unsigned cnt = 0;
    for(unsigned i = 0; i < N; ++i) {
        float a = x[i];
        if(!fast_isnan_float(a)) {
            sum_xx += a * a;
            sum_x += a;
            ++cnt;
        }
    }

    *count = cnt;
    if(cnt > 1) {
        *mean = sum_x / cnt;
        *std = sqrt(sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt));
    } else if(cnt == 1) {
        *mean = sum_x;
        *std = 0.0;
    } else {
        *mean = NAN;
        *std = NAN;
    }
}

void nan_mean_std_float_avx(float *x, unsigned N, float *mean, float *std, unsigned *count) {
    unsigned BOUNDARY = (N / 8) * 8;
    unsigned index = 0;
    float sum_xx = 0.0, sum_x = 0.0;
    unsigned cnt = 0;
    __m256 sum_xx_reg = _mm256_set1_ps(0.0);
    __m256 sum_x_reg = _mm256_set1_ps(0.0);
    __m256 cnt_reg = _mm256_set1_ps(0.0);
    __m256 all_ones = _mm256_set1_ps(1.0);
    while(index < BOUNDARY) {
        __m256 a = _mm256_loadu_ps(x + index);
        __m256 mask = _mm256_cmp_ps(a, a, _CMP_EQ_OQ);
        a = _mm256_and_ps(mask, a);
        sum_xx_reg = _mm256_fmadd_ps(a, a, sum_xx_reg);
        sum_x_reg = _mm256_add_ps(a, sum_x_reg);
        cnt_reg = _mm256_add_ps(_mm256_and_ps(mask, all_ones), cnt_reg);
        index += 8;
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, sum_xx_reg);
    sum_xx = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, sum_x_reg);
    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
    _mm256_storeu_ps(buffer, cnt_reg);
    cnt = (unsigned)(round(buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7]));

    while(index < N) {
        float a = x[index];
        if(!fast_isnan_float(a)) {
            sum_xx += a * a;
            sum_x += a;
            ++cnt;
        }
        ++index;
    }

    *count = cnt;
    if(cnt > 1) {
        *mean = sum_x / cnt;
        *std = sqrt(sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt));
    } else if(cnt == 1) {
        *mean = sum_x;
        *std = 0.0;
    } else  {
        *mean = NAN;
        *std = NAN;
    }
}

void nan_mean_std_double(double *x, unsigned N, double *mean, double *std, unsigned *count) {
    double sum_xx = 0.0, sum_x = 0.0;
    unsigned cnt = 0;
    for(unsigned i = 0; i < N; ++i) {
        double a = x[i];
        if(!fast_isnan_double(a)) {
            sum_xx += a * a;
            sum_x += a;
            ++cnt;
        }
    }

    *count = cnt;
    if(cnt > 1) {
        *mean = sum_x / cnt;
        *std = sqrt(sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt));
    } else if(cnt == 1) {
        *mean = sum_x;
        *std = 0.0;
    } else {
        *mean = NAN;
        *std = NAN;
    }
}

void nan_mean_std_double_avx(double *x, unsigned N, double *mean, double *std, unsigned *count) {
    unsigned BOUNDARY = (N / 4) * 4;
    unsigned index = 0;
    double sum_xx = 0.0, sum_x = 0.0;
    unsigned cnt = 0;
    __m256d sum_xx_reg = _mm256_set1_pd(0.0);
    __m256d sum_x_reg = _mm256_set1_pd(0.0);
    __m256d cnt_reg = _mm256_set1_pd(0.0);
    __m256d all_ones = _mm256_set1_pd(1.0);
    while(index < BOUNDARY) {
        __m256d a = _mm256_loadu_pd(x + index);
        __m256d mask = _mm256_cmp_pd(a, a, _CMP_EQ_OQ);
        a = _mm256_and_pd(mask, a);
        sum_xx_reg = _mm256_fmadd_pd(a, a, sum_xx_reg);
        sum_x_reg = _mm256_add_pd(a, sum_x_reg);
        cnt_reg = _mm256_add_pd(_mm256_and_pd(mask, all_ones), cnt_reg);
        index += 4;
    }
    double buffer[4];
    _mm256_storeu_pd(buffer, sum_xx_reg);
    sum_xx = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, sum_x_reg);
    sum_x = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    _mm256_storeu_pd(buffer, cnt_reg);
    cnt = (unsigned)(round(buffer[0] + buffer[1] + buffer[2] + buffer[3]));

    while(index < N) {
        double a = x[index];
        if(!fast_isnan_double(a)) {
            sum_xx += a * a;
            sum_x += a;
            ++cnt;
        }
        ++index;
    }

    *count = cnt;
    if(cnt > 1) {
        *mean = sum_x / cnt;
        *std = sqrt(sum_xx / cnt - (sum_x / cnt) * (sum_x / cnt));
    } else if(cnt == 1) {
        *mean = sum_x;
        *std = 0.0;
    } else  {
        *mean = NAN;
        *std = NAN;
    }
}

