import json
import numpy as np
from astropy.io import fits
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt
import seaborn as sns


slitgroups = {
    "A": ["p79", "p80", "p81", "p82"],
    "B": ["j65", "j66", "j67", "j68"],
    "C": ["j69", "j70", "j71"],
    "D": ["jw72", "jw73", "jw74"],
    "E": ["je75", "je76"],
}
# dictionary to hold fit results (best-fit values and covariance array)
fit_results = {group: {} for group in slitgroups}


# wavelength region to extract (up to He I line)
islice = slice(121, 550)

# wavelength regions to use for continuum fit
continuum_islices = slice(7, 90), slice(92, 220), slice(380, 447), slice(480, 550)

# wavelength regions to use for line fit
absfit_islices = slice(225, 358), slice(382, 410)

# Heliocentric velocity of OMC
vel_omc = +9.0 + 19.1

# Heliocentric correction
helio_topo_correction = {
    "A": -4.0,
    "B": -3.49,
    "C": -3.45,
    "D": -3.41,
    "E": -3.39,
}

light_speed = 2.99792458e5 


# For fitting the absorption line
fitter = fitting.LevMarLSQFitter()

fig, ax = plt.subplots(figsize=(6, 4))

offset = 0.0
for group, slits in slitgroups.items():
    # Find an average spectrum for each group of slits
    groupwavs = None
    groupspec = None

    for slit in slits:
        fn = f"Extract/{slit}b-cr-order53.fits"
        hdulist = fits.open(fn)

        # region to extract along slit 
        if slit in ["p79"]:
            # Full length of 14 arcsec slits
            jslice = slice(12, 49)
        else:
            # Only 23 pixels = 11 arcsec for central portion of 28 arcsec slits
            jslice = slice(18, 46)

        # Take mean over slice along slit length
        spec = hdulist["SCI"].data[jslice, :].mean(axis=0)
        wavs = hdulist["WAV"].data[jslice, :].mean(axis=0)

        # Remove bad columns
        spec[90:92] = np.nan
        # Construct clean continuum regions
        cspec = np.concatenate([spec[_] for _ in continuum_islices])
        cwavs = np.concatenate([wavs[_] for _ in continuum_islices])

        # Fit polynomial to continuum
        p = np.poly1d(np.polyfit(cwavs, cspec, 50))

        # normalize by continuum fit
        spec /= p(wavs)

        # Add in to group
        if groupwavs is None:
            groupwavs = wavs
            groupspec = spec
        else:
            groupspec += spec


    # Divide by number of contributing slits to get average
    groupspec /= len(slits)

    # Put wavelengths in frame of OMC
    groupwavs *= (1.0  - (vel_omc + helio_topo_correction[group])/light_speed)

    # Construct regions for absorption line fitting
    aspec = np.concatenate([groupspec[_] for _ in absfit_islices])
    awavs = np.concatenate([groupwavs[_] for _ in absfit_islices])

    # Fit GaussianAbsorption1D to the absorption line
    g_init = Gaussian1D(amplitude=0.1, mean=6664.0, stddev=1.0)
    g = fitter(g_init, awavs, 1.0 - aspec)

    fit_results[group]["slits"] = slits
    fit_results[group]["parameters"] = list(g.parameters)
    fit_results[group]["param err"] = np.sqrt(np.diag(
        fitter.fit_info['param_cov'])).tolist()
    fit_results[group]["covariance"] = fitter.fit_info["param_cov"].tolist()
    signal_to_noise = fit_results[group]["parameters"][0] / fit_results[group]["param err"][0]
    fit_results[group]["S/N"] = signal_to_noise

    # Add to plot
    ax.plot(groupwavs[islice], groupspec[islice] + offset, label=group, lw=0.7)
    if signal_to_noise > 5.0:
        ax.plot(groupwavs[islice], 1.0 - g(groupwavs[islice]) + offset,
                label="_nolabel_", lw=0.7, color="k")
    ax.axhline(1.0 + offset, 0.03, 0.97, color="k", lw=0.2)
    ax.annotate(group, (groupwavs[islice][-1], 1.0 + offset), xytext=(8, 0), textcoords="offset points", ha="left", va="center")
    offset += 0.1


ni2_vels = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
ni2_wavs = 6666.80 * (1.0 + ni2_vels/light_speed)
ni2_y0 = 1.9
annot_kwds = dict(textcoords="offset points", fontsize=6)
ax.plot(ni2_wavs, [ni2_y0]*5, "k|", ms=3)
ax.plot(ni2_wavs, [ni2_y0]*5, "k-", lw=0.4)
ax.axvline(ni2_wavs[2], 0.25, 0.9, color="k", lw=0.2)
ax.annotate("$-20$", (ni2_wavs[0], ni2_y0), xytext=(2, 4), ha="right", **annot_kwds)
ax.annotate("$0$", (ni2_wavs[2], ni2_y0), xytext=(0, 4), ha="center", **annot_kwds)
ax.annotate("$20$ km/s", (ni2_wavs[-1], ni2_y0), xytext=(-2, 4), ha="left", **annot_kwds)
ax.annotate("[Ni Ⅱ] 6666.8 Å", (ni2_wavs[2], ni2_y0), xytext=(0, 12), ha="center", **annot_kwds)

ax.annotate("[N Ⅱ] 6548 Å\nInter-order bleed", (6671.5, 1.6), xytext=(0, 4), ha="center", **annot_kwds)

o1_vels = np.array([-20.0, -10.0, 0.0, 10, 20.0])
o1_wavs = 6663.7473 * (1.0 + 6.3999329*o1_vels/light_speed)
o1_y0 = 0.85
annot_kwds = dict(textcoords="offset points", fontsize=7, va="top")
ax.plot(o1_wavs, [o1_y0]*5, "k|", ms=3)
ax.plot(o1_wavs, [o1_y0]*5, "k-", lw=0.4)
ax.axvline(o1_wavs[2], 0.19, 0.65, color="k", lw=0.2)
ax.annotate("$-20$", (o1_wavs[0], o1_y0), xytext=(4, -6), ha="right", **annot_kwds)
ax.annotate("$0$", (o1_wavs[2], o1_y0), xytext=(0, -6), ha="center", **annot_kwds)
ax.annotate("$20$ km/s", (o1_wavs[-1], o1_y0), xytext=(-4, -6), ha="left", **annot_kwds)
ax.annotate("O Ⅰ ${}^{3}P_{0} \\to {}^{3}D_{1}$ 1028.1571 Å\nRaman-scattered Lyβ → Hα", (o1_wavs[2], o1_y0), xytext=(0, -16), ha="center", **annot_kwds)

# ax.legend(fontsize="x-small")
ax.set(
    xlabel = "STP wavelength, Å (OMC frame)",
    ylabel = "Relative intensity",
    ylim=[0.55, 2.05],
)
ax.minorticks_on()
sns.despine()
fig.tight_layout()

figfile = "order51-absorption-by-group.pdf"
fig.savefig(figfile)

with open(figfile.replace(".pdf", ".json"), "w") as f:
    json.dump(fit_results, f, indent=4)
