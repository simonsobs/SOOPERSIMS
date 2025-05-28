import os
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
from matplotlib import cm


def Krj_Kcmb(nu):
    """
    Convert from Rayleigh-Jeans to thermodynamic (CMB) units,
    at frequency nu.
    -----------------
    input:
        nu: frequency
    output:
        RJ -> CMB conversion factor
    """
    nu = np.array(nu, dtype=np.float64)
    x = nu / 56.78
    return (np.exp(x) - 1)**2. / (x**2. * np.exp(x))


def fcmb(nu):
    """
    Convert from thermodynamic (CMB) to Rayleigh-Jeans units,
    at frequency nu.
    Inverse operation of Krj_Kcmb().
    -----------------
    input:
        nu: frequency
    output:
        CMB -> RJ conversion factor
    """
    x = 0.017608676067552197*nu
    ex = np.exp(x)
    return ex * (x / (ex-1))**2


def load_mask(config_mask, nside):
    """
    Load apodized mask from custom path given in paramfile.
    If path not specified, it will load default SAT MSS2 mask.
    -----------------
    input:
        config_mask: mask file path or None
        nside: pixel resolution
    output:
        apodized mask, degraded to nside
    """
    print("-------------------")
    print("loading apodized mask")
    if config_mask is None:
        mask_dir = "/global/cfs/cdirs/sobs/awg_bb/masks/mask_apo10.0_MSS2_SAT1_f090_coadd.fits"
    else:
        mask_dir = config_mask
    mask = hp.read_map(mask_dir)
    mask = hp.ud_grade(mask, nside)
    return mask


def compute_cls(maps, msk, wsp):
    """
    Compute auto power spectra of maps on a given mask,
    decoupled with a given mode-coupling matrix.
    -----------------
    input:
        maps: healpy maps [IQU] to compute the auto-spectra of
        msk: apodized mask on which spectra will be computed
        wsp: list of workspaces containing the mode-coupling matrices
             [spin0xspin0, spin2xspin2]
    output:
        cells: decoupled power spectra [TT, EE, EB, BE, BB]
    """
    field_0 = nmt.NmtField(msk, [maps[0]])  # I
    field_2 = nmt.NmtField(msk, [maps[1], maps[2]])  # Q,U
    cl_0x0 = wsp[0].decouple_cell(nmt.compute_coupled_cell(field_0, field_0))
    cl_2x2 = wsp[1].decouple_cell(nmt.compute_coupled_cell(field_2, field_2))
    cl_2x2 = np.array([cl_2x2[0], cl_2x2[-1]])
    cells = np.concatenate((cl_0x0, cl_2x2), axis=0)
    return cells


def plaw(x, A, alpha):
    """
    Power law model.
    -----------------
    """
    return A * (x / 80.)**alpha


def plaw_log(x, A, alpha):
    """
    Log of the power law model, for the fitting routine.
    -----------------
    """
    return np.log10(A) + alpha*x - alpha*np.log10(80.)


def fit_routine(xdata, ydata):
    """
    Fit function, it fits data in a range of good-quality data
    and provides the best-fit parameters.
    Linear fit in log-scale.
    -----------------
    input:
        x: full vector of independent variable
        y: full vector of dependent variable
           in the form y = [cl_tt, cl_ee, cl_bb]
    output:
        A: list of fitted power law amplitudes
        alpha: list of fitted power law slopes
        idx_min: starting index for the fit
        idx_max: last index for the fit
    """
    from scipy.optimize import curve_fit
    y = np.log10(ydata)
    idx_min = []
    idx_max = -3
    amplitudes = []
    slopes = []
    for i in range(y.shape[0]):
        # for the fit, select range of data after nan
        # idx_min is the index after the last nan value
        idx_min.append(np.where(np.isnan(y[i][:10]))[0][-1] + 1)
        y_fit = y[i][idx_min[i]:idx_max]
        x_fit = np.log10(xdata[idx_min[i]:idx_max])
        (A, alpha), _ = curve_fit(plaw_log, x_fit, y_fit)
        amplitudes.append(A)
        slopes.append(alpha)
    amplitudes = list(map(float, amplitudes))
    slopes = list(map(float, slopes))
    idx_min = list(map(int, idx_min))
    return amplitudes, slopes, idx_min, idx_max


def comp_sed(nu, nu0, beta, temp, typ):
    """
    Takes as input an amplitude in CMB units,
    return the amplitude extrapolated at frequency nu, in RJ units.
    -----------------
    input:
        nu: frequency to scale to
        nu0: reference frequency for the component
        beta: SED power law index (e.g., beta_synch or beta_dust)
        typ: name of the component, can be 'cmb', 'dust' or 'synch'
    output:
        frequency scaling factor
    """
    if typ == 'cmb':
        return fcmb(nu)
    elif typ == 'dust':
        x_to = 0.04799244662211351 * nu / temp
        x_from = 0.04799244662211351 * nu0 / temp
        ex_to = np.exp(x_to)
        ex_from = np.exp(x_from)
        return (nu/nu0)**(1+beta) * (ex_from-1)/(ex_to-1) * fcmb(nu0)
    elif typ == 'synch':
        return (nu/nu0)**beta * fcmb(nu0)
    return None


def plotter_templates(plots_dir, maps_synch, maps_dust,
                      maps_synch_dg_rot, maps_dust_dg_rot,
                      mask, cls_s, cls_d,
                      A_synch, A_dust, alpha_synch, alpha_dust,
                      fit_min_s, fit_min_d, fit_max, ell_arr, ell_0,
                      ):
    """
    plot results of templates_fit.py
    -----------------
    input:
        plots_dir: str, path where to save plots
        maps_{comp}: original templates maps, in K_CMB units
        maps_{comp}_dg_rot: templates maps, degraded and rotated
        mask: analysis mask for spectra computation
        cls_{c}: array, power spectra
        A_{comp}: list, fitted power-law amplitudes
        alpha_{comp}: list, fitted power-law slopes
        fit_min, fit_max: index range of spectral fit, output of 'fit_routine'
        ell_arr: array of effective ells
        ell_0: int, pivot ell
    output:
        plots
    """
    stokes = ['I', 'Q', 'U']
    vrange_synch = [1e3, 1e2]
    vrange_dust = [5e3, 5e2]

    print("plotting templates")
    plt.figure(figsize=(12, 6))
    plt.suptitle("PySM templates d9s4")
    for i in range(3):
        v = min(i, 1)
        hp.projview(maps_synch[i], sub=(2, 3, (i+4)),
                    cmap=cm.viridis, min=-vrange_synch[v] if i != 0 else -100,
                    max=vrange_synch[v], unit=r'$\mu K_{CMB}$',
                    rlabel=stokes[i], llabel='synch' if i == 0 else '')
        hp.projview(maps_dust[i], sub=(2, 3, (i+1)),
                    cmap=cm.inferno, min=-vrange_dust[v],
                    max=vrange_dust[v], unit=r'$\mu K_{CMB}$',
                    rlabel=stokes[i], llabel='dust' if i == 0 else '')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/templates.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.suptitle("PySM templates d9s4 - dgraded, rotated")
    for i in range(3):
        v = min(i, 1)
        hp.projview(maps_synch_dg_rot[i], sub=(2, 3, (i+4)),
                    cmap=cm.viridis, min=-vrange_synch[v] if i != 0 else -100,
                    max=vrange_synch[v], unit=r'$\mu K_{CMB}$',
                    rlabel=stokes[i], llabel='synch' if i == 0 else '')
        hp.projview(maps_dust_dg_rot[i], sub=(2, 3, (i+1)),
                    cmap=cm.inferno, min=-vrange_dust[v],
                    max=vrange_dust[v], unit=r'$\mu K_{CMB}$',
                    rlabel=stokes[i], llabel='dust' if i == 0 else '')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/templates_dg_rot.png", bbox_inches='tight')
    plt.close()

    print("-------------------")
    print("plotting mask")
    hp.projview(mask, title="apodized mask")
    plt.savefig(f"{plots_dir}/mask.png", bbox_inches='tight')
    plt.close()

    print("-------------------")
    print("plotting some spectra")
    # some dictionaries for plotting
    fgs_names = ['synchro', 'dust']
    cls_fgs = {fgs_names[0]: cls_s, fgs_names[1]: cls_d}
    A = {fgs_names[0]: A_synch, fgs_names[1]: A_dust}
    alpha = {fgs_names[0]: alpha_synch, fgs_names[1]: alpha_dust}
    fit_min = {fgs_names[0]: fit_min_s, fgs_names[1]: fit_min_d}
    fields = ['TT', 'EE', 'BB']
    colors = ['k', 'r', 'b']
    fmt = ['k--', 'r--', 'b--']
    xlabel = r"multipole $\ell$"
    ylabel = r"$C_\ell \, [\mu K^2]$"
    figsize_spectra = (10, 4)
    font_labels = 14

    print("plotting full spectra")
    fig, axs = plt.subplots(1, 2, figsize=figsize_spectra,
                            sharey=True, layout='constrained')
    for j, ax in enumerate(axs):
        for i in range(3):
            fgs = fgs_names[j]
            cl = cls_fgs[fgs][i]
            ax.plot(ell_arr, cl, colors[i], label=fields[i])
            ax.loglog()
            ax.set_title(fgs)
            ax.set_xlabel(xlabel, fontsize=font_labels)
        ax.axvline(ell_0, c='c', ls=':',
                   alpha=0.75, label=fr'$\ell={ell_0:d}$')
    axs[0].set_ylabel(ylabel, fontsize=font_labels)
    plt.legend()
    plt.savefig(f"{plots_dir}/spectra.png", bbox_inches='tight')
    plt.close()

    print("plotting selected data for the fit")
    fig, axs = plt.subplots(1, 2, figsize=figsize_spectra,
                            sharey=True, layout='constrained')
    for j, ax in enumerate(axs):
        for i in range(3):
            fgs = fgs_names[j]
            ells = ell_arr[fit_min[fgs][i]:fit_max]
            cl = cls_fgs[fgs][i]
            ax.plot(ells, cl[fit_min[fgs][i]:fit_max],
                    colors[i], label=fields[i])
            ax.loglog()
            ax.set_title(fgs)
            ax.set_xlabel(xlabel, fontsize=font_labels)
        ax.axvline(ell_0, c='c', ls=':',
                   alpha=0.75, label=fr'$\ell={ell_0:d}$')
    axs[0].set_ylabel(ylabel, fontsize=font_labels)
    plt.legend()
    plt.savefig(f"{plots_dir}/fit_data.png", bbox_inches='tight')
    plt.close()

    print("plotting fit results")
    lmin = 0
    ells = ell_arr[lmin:]
    fig, axs = plt.subplots(1, 2, figsize=figsize_spectra,
                            sharey=True, layout='constrained')
    for j, ax in enumerate(axs):
        for i in range(3):
            fgs = fgs_names[j]
            cl = cls_fgs[fgs][i][lmin:]
            ax.plot(ells, plaw(ells, A[fgs][i], alpha[fgs][i]), fmt[i])
            ax.plot(ells, cl, colors[i], alpha=0.65, label=fields[i])
            ax.loglog()
            ax.set_title(fgs)
            ax.set_xlabel(xlabel)
        ax.axvline(ell_0, c='c', ls=':',
                   alpha=0.75, label=fr'$\ell={ell_0:d}$')
    axs[0].set_ylabel(ylabel, fontsize=font_labels)
    plt.legend()
    plt.savefig(f"{plots_dir}/fit.png", bbox_inches='tight')
    plt.close()


def plotter_cov_sims(plots_dir, nside,
                     cl_cmb, cl_synch, cl_dust,
                     alm_cmb, alm_synch, alm_dust, alm,
                     seed, nu, ell_0,
                     ):
    """
    plot results of cov_sims.py
    -----------------
    input:
        plots_dir: str, path where to save plots
        nside: int, nside of the output sims
        cl_{comp}: array, power spectra (TT, EE, BB, TE)
        alm_{comp}: array, spherical harmonics coefficients
        alm: array, coadded alm (sed*cmb + sed*synch + sed*dust)
        seed: 4-zero-padded string, seed of the last sim
            to plot as example
        nu: str, frequency to plot as example
        ell_0: int, pivot ell
    output:
        plots saved in plots_dir
    """
    xlabel = r'multipole $\ell$'
    ylabel = r'$C_\ell \, [\mu K^2]$'
    font_labels = 14

    print("plotting input C_ells")
    print("- CMB")
    hp_order = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    colors = ['k', 'r', 'b', 'y']
    for i, cl in enumerate(cl_cmb[:-1]):
        plt.plot(cl, c=colors[i], label=hp_order[i])
    plt.loglog()
    plt.xlim([2, 5000])
    plt.xlabel(xlabel, fontsize=font_labels)
    plt.ylabel(ylabel, fontsize=font_labels)
    plt.title('CMB')
    plt.legend()
    plt.savefig(f"{plots_dir}/cl_cmb.png", bbox_inches='tight')
    plt.clf()

    print("- synchro")
    for i, cl in enumerate(cl_synch[:-1]):
        plt.plot(cl, c=colors[i], label=hp_order[i])
    plt.axvline(ell_0, c='c', ls=':',
                alpha=0.75, label=fr'$\ell={ell_0:d}$')
    plt.loglog()
    plt.xlim([2, 2000])
    plt.xlabel(xlabel, fontsize=font_labels)
    plt.ylabel(ylabel, fontsize=font_labels)
    plt.title('synchrotron')
    plt.legend()
    plt.savefig(f"{plots_dir}/cl_synch.png", bbox_inches='tight')
    plt.clf()

    print("- dust")
    for i, cl in enumerate(cl_dust[:-1]):
        plt.plot(cl, c=colors[i], label=hp_order[i])
    plt.axvline(ell_0, c='c', ls=':',
                alpha=0.75, label=fr'$\ell={ell_0:d}$')
    plt.loglog()
    plt.xlim([2, 2000])
    plt.xlabel(xlabel, fontsize=font_labels)
    plt.ylabel(ylabel, fontsize=font_labels)
    plt.title('dust')
    plt.legend()
    plt.savefig(f"{plots_dir}/cl_dust.png", bbox_inches='tight')
    plt.clf()

    print("-------------------")
    print("plotting an example of generated maps")
    print(f"seed {seed:04d} - {nu} GHz")
    m_cmb = hp.alm2map(alm_cmb, nside=nside)
    m_synch = hp.alm2map(alm_synch, nside=nside)
    m_dust = hp.alm2map(alm_dust, nside=nside)
    m_tot = hp.alm2map(alm, nside=nside)

    stokes = ['I', 'Q', 'U']
    units = r'$\mu \mathrm{K_{CMB}}$'
    plt.figure(figsize=(14, 14))
    for i in range(3):
        hp.projview(m_cmb[i], sub=(4, 3, (i+1)), cmap=cm.coolwarm,
                    unit=units, rlabel=stokes[i], llabel='CMB' if i == 0 else '')
        hp.projview(m_synch[i], sub=(4, 3, (i+4)), cmap=cm.viridis,
                    unit=units, rlabel=stokes[i], llabel='synch' if i == 0 else '')
        hp.projview(m_dust[i], sub=(4, 3, (i+7)), cmap=cm.inferno,
                    unit=units, rlabel=stokes[i], llabel='dust' if i == 0 else '')
        hp.projview(m_tot[i], sub=(4, 3, (i+10)), cmap=cm.coolwarm,
                    unit=units, rlabel=stokes[i], llabel='coadd' if i == 0 else '')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/sim_maps_{nu}GHz_{seed:04d}.png",
                bbox_inches='tight')
    plt.close()
