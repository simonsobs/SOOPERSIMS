import yaml
import argparse
import os
import numpy as np
import healpy as hp
import pymaster as nmt
from scipy.optimize import curve_fit


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


def get_bins(nside, edges, dell):
    """
    function for NaMaster binning scheme needed to compute spectra,
    given manual bandpowers edges.
    -----------------
    input:
        nside: nside of the maps to compute the spectra of
        edges: array with personalized bandpowers edges
        dell: bool, True if you want D_ell, False for C_ell
    output:
        bins: NmtBin object
    """
    bpws = np.zeros(3*nside, dtype=int) - 1
    weights = np.ones(3*nside)
    for ibpw, (l0, lf) in enumerate(zip(edges[:-1], edges[1:])):
        if lf < 3*nside:
            bpws[l0:lf] = ibpw

    larr_all = np.arange(3*nside)
    bins = nmt.NmtBin(nside, bpws=bpws, ells=larr_all,
                      weights=weights, is_Dell=dell)
    return bins


def compute_cls(maps, msk, wsp):
    """
    Compute auto power spectra of maps on a given mask,
    decoupled with a given mode-coupling matrix.
    -----------------
    input:
        maps: healpy maps [IQU] to compute the auto-spectra of
        msk: apodized mask on which spectra have to be computed
        wsp: list of workspaces containing the mode-coupling matrices
             [spin0xspin0, spin2xspin2]
    output:
        cells: decoupled power spectra [TT, EE, EB, BE, BB]
    """
    field_0 = nmt.NmtField(msk, [maps[0]])  # I
    field_2 = nmt.NmtField(msk, [maps[1], maps[2]])  # Q,U
    cl_0x0 = wsp[0].decouple_cell(nmt.compute_coupled_cell(field_0, field_0))
    # cl_0x2 = wsp[1].decouple_cell(nmt.compute_coupled_cell(field_0, field_2))
    cl_2x2 = wsp[1].decouple_cell(nmt.compute_coupled_cell(field_2, field_2))
    cl_2x2 = np.array([cl_2x2[0], cl_2x2[-1]])
    # cells = np.concatenate((cl_0x0, cl_0x2, cl_2x2), axis=0)
    cells = np.concatenate((cl_0x0, cl_2x2), axis=0)
    return cells


def plaw(x, A, alpha):
    """
    Power law model.
    -----------------
    """
    return A * (x / ell_0)**alpha


def plaw_log(x, A, alpha):
    """
    Log of the power law model, for the fitting routine.
    -----------------
    """
    return np.log10(A) + alpha*x - alpha*np.log10(ell_0)


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


def templates_fit(args):
    """
    Get power law spectral parameters by fitting PySM foregrounds templates.
    Read dust and synchrotron templates, downgrade, rotate and mask;
    then compute power spectra and fit parameters (amplitude and index).
    -----------------
    Inputs:
        args: yaml configuration file with parameters
    Outputs:
        update the input yaml file with fitted spectral parameters
    """

    # Load configuration file
    print("loading config file and parameters")
    fname_config = args.globals
    with open(fname_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = config['output_dir']
    plots_dir = out_dir + "/plots"
    alms_dir = out_dir + "/alm_templates"
    mcms_dir = out_dir + "/mcms"

    # create directories
    dirs = [out_dir, plots_dir, alms_dir, mcms_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Get parameters from config file
    nside_in = 2048
    nside_out = config['nside']
    npix_out = hp.nside2npix(nside_out)
    lmax_in = 3*nside_in - 1
    lmax_out = 3*nside_out - 1
    global ell_0
    ell_0 = int(config['plaw_ell_pivot'])

    # templates directories
    pysm_dir = "/global/cfs/cdirs/cmb/www/pysm-data"
    dust_dir = f"{pysm_dir}/dust_gnilc"
    synch_dir = f"{pysm_dir}/synch"

    print(f" - nside = {nside_out} -> lmax = {lmax_out}")
    print(f" - ell_pivot = {ell_0}")

    fname_synch = f"{synch_dir}/synch_template_nside{nside_in}_2023.02.25.fits"  # noqa
    fname_dust = f"{dust_dir}/gnilc_dust_template_nside{nside_in}_2023.02.10.fits"  # noqa
    ref_freqs = [23., 353.]  # synch, dust

    # read IQU; units: uK_RJ
    print("-------------------")
    print("reading templates")
    synch_rj = hp.read_map(fname_synch, field=[0, 1, 2])
    dust_rj = hp.read_map(fname_dust, field=[0, 1, 2])

    rj2cmb = Krj_Kcmb(ref_freqs)
    synch_cmb = rj2cmb[0] * synch_rj
    dust_cmb = rj2cmb[1] * dust_rj

    # free some space
    del synch_rj
    del dust_rj

    # downgrade, rotate
    print("-------------------")
    print("computing templates alm")
    fname_alm_synch = f"{alms_dir}/alm_synch_template_nside{nside_in}.npz"
    fname_alm_dust = f"{alms_dir}/alm_dust_template_nside{nside_in}.npz"
    print("- synchro")
    if not os.path.isfile(fname_alm_synch):
        alm_synch = hp.map2alm(synch_cmb, use_pixel_weights=True)
        np.savez(fname_alm_synch, alm_T=alm_synch[0], alm_E=alm_synch[1],
                 alm_B=alm_synch[2])
    else:
        print("  loading")
        alm_file = np.load(fname_alm_synch)
        alm_synch = np.array([alm_file['alm_T'], alm_file['alm_E'],
                              alm_file['alm_B']])
    print("- dust")
    if not os.path.isfile(fname_alm_dust):
        alm_dust = hp.map2alm(dust_cmb, use_pixel_weights=True)
        np.savez(fname_alm_dust, alm_T=alm_dust[0], alm_E=alm_dust[1],
                 alm_B=alm_dust[2])
    else:
        print("  loading")
        alm_file = np.load(fname_alm_dust)
        alm_dust = np.array([alm_file['alm_T'], alm_file['alm_E'],
                             alm_file['alm_B']])

    if not args.plots:
        del synch_cmb
        del dust_cmb

    print("-------------------")
    print(f"clipping to lmax={lmax_out}")
    # which alms indices to clip for downgrading
    clip_indices = []
    for m in range(lmax_out+1):
        clip_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m))  # noqa
    clip_indices = np.concatenate(clip_indices)
    # clipping alms at l>lmax_out to avoid artifacts
    alm_clip_synch = [each[clip_indices] for each in alm_synch]
    alm_clip_dust = [each[clip_indices] for each in alm_dust]

    # free some space
    del alm_synch
    del alm_dust

    print("rotating to equatorial coords")
    angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])
    hp.rotate_alm(alm_clip_synch, *angles)
    hp.rotate_alm(alm_clip_dust, *angles)

    print(f"reprojecting alm to maps at nside={nside_out}")
    synch_maps = hp.alm2map(alm_clip_synch, nside=nside_out)
    dust_maps = hp.alm2map(alm_clip_dust, nside=nside_out)

    print("-------------------")
    print("loading apodized mask")
    if config["mask"] is None:
        mask_dir = "/global/cfs/cdirs/sobs/awg_bb/masks/mask_apo10.0_MSS2_SAT1_f090_coadd.fits"
    else:
        mask_dir = config["mask"]
    mask = hp.read_map(mask_dir)
    mask = hp.ud_grade(mask, nside_out)

    print("binning")
    if config['bpw_edges'] is not None:
        bpw_edges = np.loadtxt(config['bpw_edges']).astype(int)
    else:
        ell_per_bin = 4
        bpw_edges = np.arange(2, lmax_out, ell_per_bin)
        config['bpw_edges'] = f"bandpowers of constant width ({ell_per_bin} multipoles per bin)"  # noqa
    bins = get_bins(nside_out, bpw_edges, dell=False)
    ell_arr = bins.get_effective_ells()

    print("mode coupling matrices")
    fname_mcm_0x0 = f"{mcms_dir}/mcm_0x0.fits"
    # fname_mcm_0x2 = f"{mcms_dir}/mcm_0x2.fits"
    fname_mcm_2x2 = f"{mcms_dir}/mcm_2x2.fits"
    wsp_0x0 = nmt.NmtWorkspace()
    # wsp_0x2 = nmt.NmtWorkspace()
    wsp_2x2 = nmt.NmtWorkspace()
    mdum_0 = np.zeros([1, npix_out])
    mdum_2 = np.zeros([2, npix_out])
    f_0 = nmt.NmtField(mask, mdum_0)
    f_2 = nmt.NmtField(mask, mdum_2)
    wsp_0x0.compute_coupling_matrix(f_0, f_0, bins, n_iter=3)
    # wsp_0x2.compute_coupling_matrix(f_0, f_2, bins, n_iter=3)
    wsp_2x2.compute_coupling_matrix(f_2, f_2, bins, n_iter=3)
    wsp_0x0.write_to(fname_mcm_0x0)
    # wsp_0x2.write_to(fname_mcm_0x2)
    wsp_2x2.write_to(fname_mcm_2x2)
    # workspaces = [wsp_0x0, wsp_0x2, wsp_2x2]
    workspaces = [wsp_0x0, wsp_2x2]

    print("compute power spectra")
    print("- synchro")
    cls_s = compute_cls(synch_maps, mask, workspaces)
    print("- dust")
    cls_d = compute_cls(dust_maps, mask, workspaces)

    print("-------------------")
    print("fit power spectra with power law")
    A_synch, alpha_synch, fit_min_s, fit_max = fit_routine(ell_arr, cls_s)
    A_dust, alpha_dust, fit_min_d, fit_max = fit_routine(ell_arr, cls_d)

    # update config file with fitted parameters
    config['A_synch_TT'] = A_synch[0]
    config['A_synch_EE'] = A_synch[1]
    config['A_synch_BB'] = A_synch[2]
    config['A_dust_TT'] = A_dust[0]
    config['A_dust_EE'] = A_dust[1]
    config['A_dust_BB'] = A_dust[2]
    config['alpha_synch_TT'] = alpha_synch[0]
    config['alpha_synch_EE'] = alpha_synch[1]
    config['alpha_synch_BB'] = alpha_synch[2]
    config['alpha_dust_TT'] = alpha_dust[0]
    config['alpha_dust_EE'] = alpha_dust[1]
    config['alpha_dust_BB'] = alpha_dust[2]

    # Copy the configuration file to output directory
    fit_paramfile = args.globals.replace(".yaml", "_fit.yaml")
    with open(fit_paramfile, "w") as f:
        yaml.dump(config, stream=f,
                  default_flow_style=False, sort_keys=False)

    if args.plots:
        print("-------------------")
        print("plotting...")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        stokes = ['I', 'Q', 'U']
        vrange_synch = [1e3, 1e2]
        vrange_dust = [5e3, 5e2]

        print("plotting templates")
        plt.figure(figsize=(12, 6))
        plt.suptitle("PySM templates d9s4")
        for i in range(3):
            v = min(i, 1)
            hp.mollview(synch_cmb[i], sub=(2, 3, (i+4)), cmap=cm.viridis,
                        min=-vrange_synch[v] if i != 0 else -100,
                        max=vrange_synch[v],
                        unit=r'$\mu K_{CMB}$', title=f'synch {stokes[i]}')
            hp.mollview(dust_cmb[i], sub=(2, 3, (i+1)), cmap=cm.inferno,
                        min=-vrange_dust[v],
                        max=vrange_dust[v],
                        unit=r'$\mu K_{CMB}$', title=f'dust {stokes[i]}')
        plt.savefig(f"{plots_dir}/templates.png", bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.suptitle("PySM templates d9s4 - dgraded, rotated")
        for i in range(3):
            v = min(i, 1)
            hp.mollview(synch_maps[i], sub=(2, 3, (i+4)), cmap=cm.viridis,
                        min=-vrange_synch[v] if i != 0 else -100,
                        max=vrange_synch[v],
                        unit=r'$\mu K_{CMB}$', title=f'synch {stokes[i]}')
            hp.mollview(dust_maps[i], sub=(2, 3, (i+1)), cmap=cm.inferno,
                        min=-vrange_dust[v],
                        max=vrange_dust[v],
                        unit=r'$\mu K_{CMB}$', title=f'dust {stokes[i]}')
        plt.savefig(f"{plots_dir}/templates_dg_rot.png", bbox_inches='tight')
        plt.close()

        print("-------------------")
        print("plotting mask")
        hp.mollview(mask, title="apodized mask")
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

        print("plotting full spectra")
        lmin = 0
        ells = ell_arr[lmin:]
        fig, axs = plt.subplots(1, 2, figsize=figsize_spectra,
                                sharey=True, layout='constrained')
        for j, ax in enumerate(axs):
            for i in range(3):
                fgs = fgs_names[j]
                cl = cls_fgs[fgs][i]
                ax.plot(ells, cl[lmin:], colors[i], label=fields[i])
                ax.loglog()
                ax.set_title(fgs)
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
            ax.axvline(ell_0, c='c', ls=':',
                       alpha=0.5, label=fr'$\ell={ell_0:d}$')
        plt.legend(frameon=False)
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
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
            ax.axvline(ell_0, c='c', ls=':',
                       alpha=0.5, label=fr'$\ell={ell_0:d}$')
        plt.legend(frameon=False)
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
                if j == 0:
                    ax.set_ylabel(ylabel)
            ax.axvline(ell_0, c='c', ls=':',
                       alpha=0.5, label=fr'$\ell={ell_0:d}$')
        plt.legend(frameon=False)
        plt.savefig(f"{plots_dir}/fit.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for transfer function")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    templates_fit(args)
