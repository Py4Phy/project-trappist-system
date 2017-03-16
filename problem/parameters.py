# ASU PHY 494 Project 2: Data for the TRAPPIST-1 system
# Copyright (c) Oliver Beckstein and Ian Kenney 2017
# Released under the MIT License

import numpy as np

solar_system = {
    'earth':  {
        'mass':5.972e24,   # in kg
        'radius': 6.371e3  # in km
    },
    'sun': {
        'mass':1.989e30,     # in kg
        'radius': 695.8e3    # in km
    }
}

astronomical_unit = 149597870.700  # in km
au = astronomical_unit

# units
# - mass: in star masses (TRAPPIST-1)
# - period: in earth days (1 yr = 365.25 d)
# - radius: in earth radii
# - semi-major: in 1e-3 au (e.g., 11.11 -> 11.11e-3 au)
# - eccentricity: unit-less
planets = {
    'name': np.array(['b', 'c', 'd', 'e', 'f', 'g']),
    'mass': np.array([3.18221540e-05, 5.16642030e-05, 1.53495096e-05,
                      2.32114535e-05, 2.54577232e-05, 5.01666899e-05]),
    'period': np.array([1.51087081, 2.4218233, 4.04961,
                        6.099615,   9.20669,  12.35294]),
    'radius': np.array([1.086,  1.056,  0.772,
                        0.918,  1.045,  1.127]),
    'semi-major': np.array([ 11.11,  15.21,  21.44,
                             28.17,  37.1 ,  45.1 ]),
    'eccentricity': np.array([ 0.081,  0.083,  0.07 ,
                               0.085,  0.063,  0.061])
}

star = {
    'name': 'TRAPPIST-1',
    'mass': 1.0,            # in star masses
    'mass_solar': 0.0802,   # in solar masses
    'radius_solar': 0.117,  # in solar radii
}


# mass of TRAPPIST-1 in solar masses
M_star_in_solar_mass = star['mass_solar']
# radius of TRAPPIST-1 in 1e-3 au
# 0.544182879201903  10^-3 au
star_radius_localunits = star['radius_solar'] * solar_system['sun']['radius'] / \
                         (astronomical_unit * 1e-3)


# in AU (solar system) units:
# GM = 4pi^2 AU^3/year^2
#G = 4*np.pi**2

# Gravitational constant in TRAPPIST-1 local units
# length = 1e-3 AU, time = d, M = M_star
G_local = 4*np.pi**2 * (1e3)**3 / (365.25)**2 * M_star_in_solar_mass
