# PorePy-Reaktoro-coupling

This repostory contains the run scripts for the paper "A discrete fracture matrix framework for simulating single-phase flow and non-isothermal reactive transport". The implementation is by Shin Irgens Banshoya, PhD at Dept. of Mathematics, UiB. The simulations used commit cabb56e.

The implementation uses the banch ´modified_discretization_matrices' of PorePy. The easiest way to install this branch is to follow the PorePy installation instructions, but replace the line `git checkout main` to `git checkout modified_discretization_matrices`. Also version 2.8.0 of Reaktoro is used.

Please be aware that the 2D simulations will take some time run. Running these cases on your own computer is at our own risk.
