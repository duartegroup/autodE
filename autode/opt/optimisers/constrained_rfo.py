"""
Perform a rational function optimisation in delocalised internal coordinates
(DIC) to a minimum under a set of distance constraints. Requires:

1. Forming a set of DIC that contains the distances as primitives (as well)
   as the other completely redundant set
2. Schmidt-orthogonalsing the resultant U matrix
3. Remove primitives coordinates with zero weight
4. Form the Lagrangian function
5. Modify the Hessian matrix with

Refs.

[1] J. Baker, A. Kessi and B. Delley, J. Chem. Phys. 105, 1996, 192
[2] J. Baker, J. Comput. Chem. 18, 1997, 1079
[3] https://manual.q-chem.com/5.2/A1.S5.html
"""






