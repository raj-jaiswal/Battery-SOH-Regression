#include <math.h>
static inline double pdiv(double a,double b){return fabs(b)>1e-3?a/b:1.0;}
static inline double psqrt(double a){return sqrt(fabs(a));}
static inline double plog(double a){return fabs(a)>1e-3?log(fabs(a)):0.0;}
// Returns standardized residual correction; GP output is also in standardized SOH units.
double wd_gpr_sr_residual(const double *x){return (x[0] * -0.03718976144);}
