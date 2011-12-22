cdef extern from "gsl/gsl_sf_fermi_dirac.h":

  double  gsl_sf_fermi_dirac_m1(double x) nogil

  int  gsl_sf_fermi_dirac_m1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_0(double x) nogil

  int  gsl_sf_fermi_dirac_0_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_1(double x) nogil

  int  gsl_sf_fermi_dirac_1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_2(double x) nogil

  int  gsl_sf_fermi_dirac_2_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_int(int j, double x) nogil

  int  gsl_sf_fermi_dirac_int_e(int j, double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_mhalf(double x) nogil

  int  gsl_sf_fermi_dirac_mhalf_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_half(double x) nogil

  int  gsl_sf_fermi_dirac_half_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_3half(double x) nogil

  int  gsl_sf_fermi_dirac_3half_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_fermi_dirac_inc_0(double x, double b) nogil

  int  gsl_sf_fermi_dirac_inc_0_e(double x, double b, gsl_sf_result * result) nogil

