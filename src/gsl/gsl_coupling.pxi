cdef extern from "gsl/gsl_sf_coupling.h":

  double  gsl_sf_coupling_3j(int two_ja, int two_jb, int two_jc, int two_ma, int two_mb, int two_mc) nogil

  int  gsl_sf_coupling_3j_e(int two_ja, int two_jb, int two_jc, int two_ma, int two_mb, int two_mc, gsl_sf_result * result) nogil

  double  gsl_sf_coupling_6j(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf) nogil

  int  gsl_sf_coupling_6j_e(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf, gsl_sf_result * result) nogil

  double  gsl_sf_coupling_9j(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf, int two_jg, int two_jh, int two_ji) nogil

  int  gsl_sf_coupling_9j_e(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf, int two_jg, int two_jh, int two_ji, gsl_sf_result * result) nogil

