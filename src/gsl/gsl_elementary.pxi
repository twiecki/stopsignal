cdef extern from "gsl/gsl_sf_elementary.h":

  int  gsl_sf_multiply_e(double x, double y, gsl_sf_result * result) nogil

  int  gsl_sf_multiply_err_e(double x, double dx, double y, double dy, gsl_sf_result * result) nogil

