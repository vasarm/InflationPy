from pathlib import Path

import numpy as np

from inflationpy.data.data import load_planck_data


def observation_plot(ax, label_bool=True, sigma1_label="1σ rogion", sigma2_label="2σ region"):
    """Take in matplotlib axis and plot planck 2018 observation regions on.

    Parameters
    ----------
    ax : _type_
        _description_
    label_bool : bool, optional
        Plot with labels boolean, by default True
    sigma1_label : str, optional
        _description_, by default '1σ rogion'
    sigma2_label : str, optional
        _description_, by default '2σ region'
    """

    sigma1_ns, sigma1_r, sigma2_ns, sigma2_r = load_planck_data()

    if label_bool:
        ax.plot(sigma1_ns, sigma1_r, label=sigma1_label)
        ax.plot(sigma2_ns, sigma2_r, label=sigma2_label)
    else:
        ax.plot(sigma1_ns, sigma1_r)
        ax.plot(sigma2_ns, sigma2_r)


def ns_r_plot(ns, r, ax, label_bool=True, label="Model"):
    if label_bool:
        ax.plot(ns, r, label=label)
    else:
        ax.plot(ns, r)


def full_ns_r_plot(ns, r, ax, label_bool, label="Model", sigma1_label="1σ rogion", sigma2_label="2σ region"):
    observation_plot(ax, label_bool, sigma1_label, sigma2_label)
    ns_r_plot(ns, r, ax, label_bool, label)
