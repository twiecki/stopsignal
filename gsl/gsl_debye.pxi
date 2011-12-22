cdef extern from "gsl/gsl_sf_debye.h":

  double  gsl_sf_debye_1(double x) nogil

  int  gsl_sf_debye_1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_debye_2(double x) nogil

  int  gsl_sf_debye_2_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_debye_3(double x) nogil

  int  gsl_sf_debye_3_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_debye_4(double x) nogil

  int  gsl_sf_debye_4_e(double x, gsl_sf_result * result) nogil

