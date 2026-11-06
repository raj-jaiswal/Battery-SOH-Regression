#include <math.h>

static inline double pdiv(double a, double b) { return fabs(b) > 0.001 ? a / b : 1.0; }
static inline double psqrt(double a) { return sqrt(fabs(a)); }
static inline double plog(double a) { return fabs(a) > 0.001 ? log(fabs(a)) : 0.0; }

// Returns standardized SOH; apply target_scale and target_mean from metadata.json.
double battery_symbolic_predict(const double *x) { return (x[6] * psqrt(((-0.73534506408587941) + x[20]))); }
