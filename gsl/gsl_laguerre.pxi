cdef extern from "gsl/gsl_sf_laguerre.h":

  double  gsl_sf_laguerre_1(double a, double x) nogil

  double  gsl_sf_laguerre_2(double a, double x) nogil

  double  gsl_sf_laguerre_3(double a, double x) nogil

  int  gsl_sf_laguerre_1_e(double a, double x, gsl_sf_result * result) nogil

  int  gsl_sf_laguerre_2_e(double a, double x, gsl_sf_result * result) nogil

  int  gsl_sf_laguerre_3_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_laguerre_n(int n, double a, double x) nogil

  int  gsl_sf_laguerre_n_e(int n, double a, double x, gsl_sf_result * result) nogil

