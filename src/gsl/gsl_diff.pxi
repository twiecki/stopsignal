cdef extern from "gsl/gsl_diff.h":
  int gsl_diff_central ( gsl_function *f, double x, 
                        double *result, double *abserr) nogil
  
  int gsl_diff_backward ( gsl_function *f, double x,
                         double *result, double *abserr) nogil
  
  int gsl_diff_forward ( gsl_function *f, double x,
                        double *result, double *abserr) nogil
  
