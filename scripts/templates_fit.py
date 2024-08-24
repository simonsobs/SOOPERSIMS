import yaml
import argparse
import os
import numpy as np
import healpy as hp
import pymaster as nmt
import utils as u
from scipy.optimize import curve_fit


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

    rj2cmb = u.Krj_Kcmb(ref_freqs)
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

    mask = u.load_mask(config['mask'], nside_out)

    print("binning")
    if config['bpw_edges'] is not None:
        bpw_edges = np.loadtxt(config['bpw_edges']).astype(int)
    else:
        ell_per_bin = 4
        bpw_edges = np.arange(2, lmax_out, ell_per_bin)
        config['bpw_edges'] = f"bandpowers of constant width ({ell_per_bin} multipoles per bin)"  # noqa
    bins = u.get_bins(nside_out, bpw_edges, dell=False)
    ell_arr = bins.get_effective_ells()

    print("mode coupling matrices")
    workspaces = u.compute_mcms(mcms_dir, nside_out, bins, mask)

    print("compute power spectra")
    print("- synchro")
    cls_s = u.compute_cls(synch_maps, mask, workspaces)
    print("- dust")
    cls_d = u.compute_cls(dust_maps, mask, workspaces)

    print("-------------------")
    print("fit power spectra with power law")
    A_synch, alpha_synch, fit_min_s, fit_max = u.fit_routine(ell_arr, cls_s)
    A_dust, alpha_dust, fit_min_d, fit_max = u.fit_routine(ell_arr, cls_d)

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
        u.plotter_templates(plots_dir, synch_cmb, dust_cmb,
                            synch_maps, dust_maps,
                            mask, cls_s, cls_d,
                            A_synch, A_dust, alpha_synch, alpha_dust,
                            fit_min_s, fit_min_d, fit_max, ell_arr, ell_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for transfer function")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    templates_fit(args)
