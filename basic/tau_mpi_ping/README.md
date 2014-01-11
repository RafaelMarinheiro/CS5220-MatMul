# MPI ping-pong with TAU instrumentation

The installation of the TAU system on C4 is built to support both
compile-time instrumentation (using PDT) and run-time instrumentation
(using DynInst).  For compile-time instrumentation, the software should
be build with `tau_cc.sh` (or `tau_cxx.sh` for C++ codes).  The
Makefile in this directory illustrates:

1. How to build with compile-time instrumentation 
2. How to use the `mpisub` script to run the instrumented code
3. How to view the profile results with `pprof`
4. How to pack the profile results for profiling on your own machine

