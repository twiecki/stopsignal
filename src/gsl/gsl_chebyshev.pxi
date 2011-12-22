cdef extern from "gsl/gsl_chebyshev.h":
  
  ctypedef struct gsl_cheb_series
  
  
  gsl_cheb_series * gsl_cheb_alloc( size_t order) nogil
  
  void gsl_cheb_free(gsl_cheb_series * cs) nogil
  
  int gsl_cheb_init(gsl_cheb_series * cs,  gsl_function * func,
                     double a,  double b) nogil
  
  
  double gsl_cheb_eval( gsl_cheb_series * cs,  double x) nogil
  int gsl_cheb_eval_err( gsl_cheb_series * cs,  double x, 
                        double * result, double * abserr) nogil
  
  
  double gsl_cheb_eval_n( gsl_cheb_series * cs,  size_t order, double x) nogil
  int gsl_cheb_eval_n_err( gsl_cheb_series * cs,  size_t order, 
                           double x, double * result, double * abserr) nogil
  
  
  double gsl_cheb_eval_mode( gsl_cheb_series * cs, double x, gsl_mode_t mode) nogil
  int gsl_cheb_eval_mode_e( gsl_cheb_series * cs,  double x, gsl_mode_t mode, double * result, double * abserr) nogil
  
  
  
  int gsl_cheb_calc_deriv(gsl_cheb_series * deriv,  gsl_cheb_series * cs) nogil
  
  int gsl_cheb_calc_integ(gsl_cheb_series * integ,  gsl_cheb_series * cs) nogil
  
