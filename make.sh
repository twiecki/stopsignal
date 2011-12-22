#!/bin/bash

cython stop_likelihoods.pyx

c:/Python27/Scripts/gcc.exe -mno-cygwin -mdll -O -Wall -Ic:/Python27/lib/site-packages/numpy/core/include -Ic:/Python27/include -Ic:/Python27/PC -Ic:/Program\ Files/GnuWin32/include -c stop_likelihoods.c -o stop_likelihoods.o
c:/Python27/Scripts/gcc.exe -mno-cygwin -shared -s stop_likelihoods.o -Lc:/Python27/libs -Lc:/Program\ Files/GnuWin32/lib -Lc:/Python27/PCbuild -lgsl -lpython27 -lmsvcr90 -o stop_likelihoods.pyd
 
