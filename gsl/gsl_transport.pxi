cdef extern from "gsl/gsl_sf_transport.h":

  double  gsl_sf_transport_2(double x) nogil

  int  gsl_sf_transport_2_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_transport_3(double x) nogil

  int  gsl_sf_transport_3_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_transport_4(double x) nogil

  int  gsl_sf_transport_4_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_transport_5(double x) nogil

  int  gsl_sf_transport_5_e(double x, gsl_sf_result * result) nogil

