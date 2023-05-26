`datagen`
--------

## Transition state datasets
These datasets consist of 

- `rudorff_lilienfeld`: Taken from [this](https://iopscience.iop.org/article/10.1088/2632-2153/aba822) publication. In the original paper, all calculations were performed using a different electronic structure code to QChem and do not include initial and/or final state complexes. In this work, we take _only_ the transition state structures from this publication and perform frequency calculations to ascertain if there is only one imaginary frequency. These structures are then perturbed along their eigenmode in the positive and negative direction, while keeping constraints on the outer-ring of atoms (without which dissociation of the molecule occurs, as noted in the original work). This directory contains all the python scripts required to rerun the calculations and parse the coefficient matrix from QChem calculations.
- `grambow_green`: Taken from [this](https://www.nature.com/articles/s41597-020-0460-4) work.

## Rotation datasets
These datsets consist of one type of structure rotated at different Euler angles. Largely for testing purposes.
- `rotated_water_molecules`: A single water molecule rotated at ten random Euler angles.
- `rotated_sn2_reaction`: A single S<sub>N</sub>2 reaction chosen at random from the `rudorff_lilienfeld` dataset is rotated at ten random Euler angles.