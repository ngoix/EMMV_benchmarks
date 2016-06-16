# EMMV_benchmarks
# How to evaluate unsupervised Anomaly Detection algorithms?

author: Nicolas Goix, nicolas.goix@telecom-paristech.fr

This is the code associated with ICML workshop paper

http://perso.telecom-paristech.fr/~goix/icml_workshop.pdf

File em_bench.py evaluates the algorithms using EM and MV based criteria,
without sub-sampling features nor averaging. It does not work in high dimensions.

File em_bench_high.py makes use of sub-sampling (along features) and averaging,
extending the use of these criteria to high-dimensional datasets.

Basic EM and MV calculation is implemented in em.py
