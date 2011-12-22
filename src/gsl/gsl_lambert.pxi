cdef extern from "gsl/gsl_sf_lambert.h":

  double  gsl_sf_lambert_W0(double x) nogil

  int  gsl_sf_lambert_W0_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_lambert_Wm1(double x) nogil

  int  gsl_sf_lambert_Wm1_e(double x, gsl_sf_result * result) nogil

