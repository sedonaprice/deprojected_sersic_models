import os
import numpy as np
import matplotlib.pyplot as plt
import sersic_profile_mass_VC as spm

# Environment variable containing path to location of pre-computed tables
table_dir = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR')

# Sérsic profile properties
total_mass = 1.e11
Reff = 5.0
n = 1.0
R = np.arange(0., 30.1, 0.1)

# Flattening/elongation array (invq = 1/q; q = c/a)
invq_arr = [1., 2.5, 3.33, 5., 10.]

# Calculate & plot interpolated circular velocity profiles at r for each invq
plt.figure(figsize=(4,3.5))
for invq in invq_arr:
    vc = spm.interpolate_sersic_profile_VC(R=R, total_mass=total_mass, Reff=Reff,
                                       n=n, invq=invq, path=table_dir)
    plt.plot(R, vc, '-', label=r'$q_0$={:0.2f}'.format(1./invq))

plt.xlabel('Radius [kpc]')
plt.ylabel('Circular velocity [km/s]')
plt.legend(title='Intrinsic axis ratio')

plt.tight_layout()
plt.show()