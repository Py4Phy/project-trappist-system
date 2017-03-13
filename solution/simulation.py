# ASU PHY 494 Project 2: Solution: simulation
# Copyright (c) Oliver Beckstein 2017
# All Rights Reserved.


import numpy as np

import parameters
from parameters import (M_star_in_solar_mass,  # mass of TRAPPIST-1 in solar masses
                  star_radius_localunits, # in 10^-3 au
                  )
from parameters import G_local as G_grav


def rmax(a, e):
    """Max radius of ellipitical orbit."""
    return a*(1+e)

def v_aphelion(a, e, M):
    """Velocity at aphelion (max r) in elliptical orbit."""
    return np.sqrt(G_grav*M*(1-e)/(a*(1+e)))


def F_gravity(r, m, M=1.0):
    rr = np.sum(r*r)
    rhat = r/np.sqrt(rr)
    return G_grav*m*M/rr * rhat

def U_gravity(r, m, M=1.0):
    return -G_grav*m*M/np.sqrt(np.sum(r*r, axis=-1))

def kinetic_energy(v, m):
    """Kinetic energy 1/2 m v**2"""
    # v = [[v(0)x, v(0)y], [...], ...]
    return 0.5*m*np.sum(v**2, axis=-1)

# planetary system, assuming central star to be fixed
def orbits(r0, v0, masses, M_star=1.0, dt=0.001, t_max=1):
    """2D planetary motion with velocity verlet for multiple planets.

    Central star with mass M_star is assumed to be fixed.
    """

    N_bodies = len(masses)
    assert r0.shape[0] == N_bodies
    dim = r0.shape[1]
    assert np.all(v0.shape == r0.shape)

    nsteps = int(t_max/dt)

    r = np.zeros((nsteps, N_bodies, dim))
    v = np.zeros_like(r)

    r[0, :, :] = r0
    v[0, :, :] = v0

    # start force evaluation for first step
    Ft = forces(r[0], masses, M_star)
    for i in range(nsteps-1):
        vhalf = v[i] + 0.5*dt * Ft/masses[:, np.newaxis]
        r[i+1, :] = r[i] + dt * vhalf
        Ftdt = forces(r[i+1], masses, M_star)
        v[i+1] = vhalf + 0.5*dt * Ftdt/masses[:, np.newaxis]
        # new force becomes old force
        Ft = Ftdt
    t = dt * np.arange(nsteps)
    return t, r, v

def forces(r, masses, M):
    F = np.zeros_like(r)
    for i, m in enumerate(masses):
        # planet - star (note: minus sign to be consistent with rij!)
        F[i, :] = F_gravity(-r[i], m=m, M=M)
        for j in range(i+1, len(masses)):
            # force of planet j on i
            rij = r[j] - r[i]
            Fij = F_gravity(rij, m=m, M=masses[j])
            F[i] += Fij
            # Newton's 3rd law:
            Fji = -Fij
            F[j] += Fji
    return F




def orbits2(r0, v0, masses, M_star=1.0, dt=0.001, t_max=1):
    """2D planetary motion with velocity verlet for multiple planets and star."""
    N_bodies = len(masses)
    assert r0.shape[0] == N_bodies
    dim = r0.shape[1]
    assert np.all(v0.shape == r0.shape)

    nsteps = int(t_max/dt)

    r = np.zeros((nsteps, N_bodies, dim))
    v = np.zeros_like(r)

    r[0, :, :] = r0
    v[0, :, :] = v0

    # start force evaluation for first step
    Ft = forces2(r[0], masses)
    for i in range(nsteps-1):
        vhalf = v[i] + 0.5*dt * Ft/masses[:, np.newaxis]
        r[i+1, :] = r[i] + dt * vhalf
        Ftdt = forces2(r[i+1], masses)
        v[i+1] = vhalf + 0.5*dt * Ftdt/masses[:, np.newaxis]
        # new force becomes old force
        Ft = Ftdt
        if np.linalg.norm(Ft.sum(axis=0)) > 1e-12:
            print("Violation of Newton's 3rd law: F = {}".format(Ft))

    t = dt * np.arange(nsteps)
    return t, r, v

def forces2(r, masses):
    F = np.zeros_like(r)
    for i, m in enumerate(masses):
        for j in range(i+1, len(masses)):
            # force of body j on i
            rij = r[j] - r[i]
            Fij = F_gravity(rij, m=m, M=masses[j])
            F[i] += Fij
            # Newton's 3rd law:
            Fji = -Fij
            F[j] += Fji
    return F


