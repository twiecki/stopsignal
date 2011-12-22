cdef extern from "gsl/gsl_sf_psi.h":

  double  gsl_sf_psi_int(int n) nogil

  int  gsl_sf_psi_int_e(int n, gsl_sf_result * result) nogil

  double  gsl_sf_psi(double x) nogil

  int  gsl_sf_psi_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_psi_1piy(double y) nogil

  int  gsl_sf_psi_1piy_e(double y, gsl_sf_result * result) nogil

  double  gsl_sf_psi_1_int(int n) nogil

  int  gsl_sf_psi_1_int_e(int n, gsl_sf_result * result) nogil

  double  gsl_sf_psi_n(int m, double x) nogil

  int  gsl_sf_psi_n_e(int m, double x, gsl_sf_result * result) nogil

