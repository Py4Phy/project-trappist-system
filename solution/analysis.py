# ASU PHY 494 Project 2: Solution: analysis
# Copyright (c) Oliver Beckstein 2017
# All Rights Reserved.

import numpy as np
import scipy.spatial

import matplotlib
import matplotlib.pyplot as plt


import parameters
from parameters import (M_star_in_solar_mass,  # mass of TRAPPIST-1 in solar masses
                  star_radius_localunits, # in 10^-3 au
                  )

def km2au(x):
    """Convert length x from km to au"""
    return x / parameters.astronomical_unit

def au2km(x):
    """Convert length x from au to km"""
    return x * parameters.astronomical_unit


def plot_orbits(r, ax=None):
    N_planets = r.shape[1]

    if ax is None:
        ax = plt.subplot(111)

    ax.set_aspect(1)

    for planet in range(N_planets):
        rx, ry = r[:, planet, :].T
        ax.plot(rx, ry)
        ax.plot(rx[-1], ry[-1], 'o', ms=4, color="gray", alpha=0.5)

    # star fixed at origin
    ax.plot([0], [0], 'o', ms=8, color="orange")
    return ax

def plot_orbits_fancy(r, ax=None, scale=5, star_radius=star_radius_localunits,
                      radii=None):
    N_planets = r.shape[1]

    if radii is None:
        radii = scale * system_AU['radius']*1e3
    assert len(radii) == N_planets

    if ax is None:
        ax = plt.subplot(111)
    ax.set_aspect(1)

    for planet in range(N_planets):
        rx, ry = r[:, planet, :].T
        ax.plot(rx, ry)
        ax.plot(rx[-1], ry[-1], 'o', ms=4, color="gray", alpha=0.5)

        planet_circle = plt.Circle([rx[-1], ry[-1]],
                                   radii[planet], color='gray', alpha=0.4)
        ax.add_artist(planet_circle)

    ax.add_artist(plt.Circle([0, 0], star_radius,
                             color='orange', alpha=0.9))
    return ax

def plot_orbits2(r, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    ax.set_aspect(1)

    for planet in range(r.shape[1]):
        rx, ry = r[:, planet, :].T
        ax.plot(rx, ry)
        if planet == 0:    # star
            plt.plot(rx[-1], ry[-1], 'o', color="orange", ms=8)
        else: # planet
            plt.plot(rx[-1], ry[-1], 'o', color="gray", ms=4, alpha=0.5)
    return ax

def plot_orbits_fancy2(r, ax=None, scale=5, radii=None):
    if radii is None:
        radii = scale * system_AU['radius']*1e3
    if ax is None:
        ax = plt.subplot(111)
    ax.set_aspect(1)
    for planet in range(r.shape[1]):
        rx, ry = r[:, planet, :].T
        ax.plot(rx, ry)
        if planet == 0:
            # star
            ax.add_artist(plt.Circle([rx[-1], ry[-1]],
                                     star_radius_localunits, color='orange',
                                     alpha=0.9, zorder=1))
        else: # planet
            idx = planet - 1
            radius = radii.iloc[planet]
            planet_circle = plt.Circle([rx[-1], ry[-1]], radius, color='gray',
                                       alpha=0.8, zorder=2)
            ax.add_artist(planet_circle)
    return ax



def kinetic_energy(v, m):
    """Kinetic energy 1/2 m v**2 for time step and planet"""
    # v.shape == (N_timesteps, N_planets, 2)
    return 0.5*m*np.sum(v**2, axis=-1)

def energy_conservation(t, r, v, U, **kwargs):
    """Energy drift (Tuckerman Eq 3.14.1)"""
    m = kwargs.get('m', 1)
    KE = np.sum(kinetic_energy(v, m), axis=-1)
    PE = np.sum(U(r, **kwargs), axis=-1)
    E = KE + PE

    machine_precision = 1e-15
    if np.isclose(E[0], 0, atol=machine_precision, rtol=machine_precision):
        # if E[0] == 0 then replace with machine precision (and expect bad results)
        E[0] = machine_precision
    return np.mean(np.abs(E/E[0] - 1))

def energy_precision(energy, machine_precision=1e-15):
    """log10 of relative energy conservation"""
    if np.isclose(energy[0], 0, atol=machine_precision, rtol=machine_precision):
        # if E[0] == 0 then replace with machine precision (and expect bad results)
        E = energy.copy()
        E[0] = machine_precision
    else:
        # don't modify input energies
        E = energy
    DeltaE = np.abs(E/E[0] - 1)
    # filter any zeros
    # (avoid log of zero by replacing 0 with machine precision 1e-15)
    zeros = np.isclose(DeltaE, 0, atol=machine_precision, rtol=machine_precision)
    DeltaE[zeros] = machine_precision
    return np.log10(DeltaE)

def analyze_energies(t, r, v, U, step=1, **kwargs):
    m = kwargs.get('m', 1)
    KE = np.sum(kinetic_energy(v, m=m), axis=-1)
    PE = np.sum(U(r, **kwargs), axis=-1)
    energy = KE + PE

    times = t[::step]

    ax = plt.subplot(2, 1, 1)
    ax.plot(times, KE[::step], 'r-', label="KE")
    ax.plot(times, PE[::step], 'b-', label="PE")
    ax.plot(times, energy[::step], 'k--', label="E")
    #ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend(loc="best")

    e_prec = energy_precision(energy)
    ax = plt.subplot(2, 1, 2)
    ax.plot(times, e_prec[::step])
    ax.set_xlabel("time")
    ax.set_ylabel("log(relative error energy)")

    #return ax.figure


# resolution of the naked eye in minutes
theta_eye_minutes = 1.
theta_eye = np.deg2rad(theta_eye_minutes * 1/60)

def y_resolved(d, theta=theta_eye):
    """Size of an object that can be resolved with the eye at distance d"""
    return theta_eye * d

def dmax(y, theta=theta_eye):
    """Maximum distance so that features of size y are still resolved."""
    return y/theta

def find_nearby_planets(r_planets, ymin=500):
    """Analyze number of nearby planets as a function of time."""
    cutoff = dmax(ymin)
    timeseries = np.zeros(r_planets.shape[:2])
    for i, pos in enumerate(r_planets):
        pos = pos * 1e-3  # in au; not in-place to avoid changing r_planets
        dm = au2km(scipy.spatial.distance.cdist(pos, pos)) # in km
        timeseries[i, :] = np.sum(np.logical_and(dm > 1.0, dm < cutoff), axis=1)
    return timeseries
