import numpy as np

"""This implementation is courtesy of Aaron Kaplan, Lawrence Berkeley Lab."""


def cart_to_sph_d():
    """
       From Table I of H. Bernhard Schlegel and M.J. Frisch,
       Int. J. Quantum Chem. 54, 83-87 (1995),
       doi: 10.1002/qua.560540202

       Expected order is

       Y_20                                  V_200
       Y_21_+                                V_020
    (  Y_21_-  )  = ( 5 x 6 coeff matrix ) ( V_002 )
       Y_22_+                                V_110
       Y_22_-                                V_011
                                             V_101
       where Y_lm is the **normalized** spherical harmonic with angular
       momentum l and "magnetic" quantum number m
           Y_lm_+ = (Y_lm + Y_l-m) / sqrt(2)
           Y_lm_- = (Y_lm - Y_l-m) / sqrt(2)
       to keep these real-valued

       V_lxlylz is the Cartesian angular momentum function
           x**lx * y**ly * z**lz

       This should operate on uncontracted, **normalized** cartesian gaussians
    """
    sqrt3_half = 3.0 ** (0.5) / 2.0
    cmat = np.array(
        [
            [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [sqrt3_half, -sqrt3_half, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    return cmat


def norm_cart_d_gauss(zeta):
    """
    seems to be missing factor of 2**(3/4) in Schlegel and Frisch work
    returns normalization vector to dot into same vector as before,
    [V_200, V_020, V_002, V_110, V_011, V_101],
    but V's are not normalized

    note that x**2, y**2, z**2 have same normalization, norm_200
    and xy, yz, zx have same normalization, norm_110
    """

    norm_110 = 2.0 * (8.0 * zeta**7 / np.pi**3) ** (0.25)
    norm_200 = norm_110 / 3.0 ** (0.25)
    return np.array([norm_200, norm_200, norm_200, norm_110, norm_110, norm_110])


if __name__ == "__main__":
    cmat = cart_to_sph_d()
    print(cmat, cmat.shape)
    print(norm_cart_d_gauss(1.0))
