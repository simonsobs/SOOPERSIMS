import yaml
import argparse
import os
import wget
import numpy as np
import healpy as hp
import pymaster as nmt
import utils as u


def templates_fit(args):
    """
    Get power law spectral parameters by fitting PySM foregrounds templates.
    Read dust and synchrotron templates, downgrade, and rotate;
    then evaluate power spectra on a mask and fit foregrounds
    spectral parameters (amplitude and index).
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
    plots_dir = f"{out_dir}/plots"
    templates_dir = f"{out_dir}/templates"
    mcms_dir = f"{out_dir}/mcms"

    # create directories
    dirs = [out_dir, plots_dir, templates_dir, mcms_dir]
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

    print(f" - nside = {nside_out} -> lmax = {lmax_out}")
    print(f" - ell_pivot = {ell_0}")

    # get and pre-process synchrotron and dust templates
    ref_freqs = [23., 353.]  # synch, dust
    pysm_url = "https://portal.nersc.gov/cfs/cmb/pysm-data"
    url_synch = f"{pysm_url}/synch/synch_template_nside{nside_in}_2023.02.25.fits"
    url_dust = f"{pysm_url}/dust_gnilc/gnilc_dust_template_nside{nside_in}_2023.02.10.fits"
    fname_synch = f"{templates_dir}/template_synch_maps.fits"
    fname_dust = f"{templates_dir}/template_dust_maps.fits"
    fname_alm_synch = f"{templates_dir}/template_synch_alm_lmax{lmax_in}.fits"
    fname_alm_dust = f"{templates_dir}/template_dust_alm_lmax{lmax_in}.fits"

    # if alm do not exist in the out directory
    if not os.path.isfile(fname_alm_synch) or not os.path.isfile(fname_alm_dust):
        print("-------------------")
        print("computing templates alm")
        # download templates if not present
        if not os.path.isfile(fname_synch) or not os.path.isfile(fname_dust):
            print("downloading templates")
            wget.download(url_synch, fname_synch)
            wget.download(url_dust, fname_dust)

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
        print("- synchro")
        alm_synch = hp.map2alm(synch_cmb, use_pixel_weights=True)
        # write alm to disk
        hp.write_alm(fname_alm_synch, alm_synch, overwrite=True, out_dtype=np.float64)

        print("- dust")
        alm_dust = hp.map2alm(dust_cmb, use_pixel_weights=True)
        hp.write_alm(fname_alm_dust, alm_dust, overwrite=True, out_dtype=np.float64)

        del synch_cmb
        del dust_cmb
    else:
        print("loading templates alm")
        alm_synch = hp.read_alm(fname_alm_synch, hdu=(1, 2, 3))
        alm_dust = hp.read_alm(fname_alm_dust, hdu=(1, 2, 3))

    print("-------------------")
    print(f"clipping alm to lmax={lmax_out}")
    alm_clip_synch = hp.resize_alm(alm_synch, lmax_in, lmax_in, lmax_out, lmax_out)
    alm_clip_dust = hp.resize_alm(alm_dust, lmax_in, lmax_in, lmax_out, lmax_out)

    maps_synch_dg = hp.alm2map(alm_clip_synch, nside=nside_out)
    maps_dust_dg = hp.alm2map(alm_clip_dust, nside=nside_out)

    # free some space
    del alm_synch
    del alm_dust

    print("rotating to equatorial coords")
    angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])
    hp.rotate_alm(alm_clip_synch, *angles)
    hp.rotate_alm(alm_clip_dust, *angles)

    print(f"reprojecting alm to maps at nside={nside_out}")
    maps_synch = hp.alm2map(alm_clip_synch, nside=nside_out)
    maps_dust = hp.alm2map(alm_clip_dust, nside=nside_out)

    mask = u.load_mask(config['mask'], nside_out)

    print("binning")
    delta_ell = 4
    bins = nmt.NmtBin.from_lmax_linear(lmax_out, nlb=delta_ell)
    ell_arr = bins.get_effective_ells()

    print("mode coupling matrices")
    fname_mcm_0x0 = f"{mcms_dir}/mcm_0x0.fits"
    fname_mcm_2x2 = f"{mcms_dir}/mcm_2x2.fits"
    f0 = nmt.NmtField(mask, None, spin=0)
    f2 = nmt.NmtField(mask, None, spin=2)
    wsp_0x0 = nmt.NmtWorkspace.from_fields(f0, f0, bins)
    wsp_2x2 = nmt.NmtWorkspace.from_fields(f2, f2, bins)
    wsp_0x0.write_to(fname_mcm_0x0)
    wsp_2x2.write_to(fname_mcm_2x2)
    workspaces = [wsp_0x0, wsp_2x2]

    print("compute power spectra")
    print("- synchro")
    cls_s = u.compute_cls(maps_synch, mask, workspaces)
    print("- dust")
    cls_d = u.compute_cls(maps_dust, mask, workspaces)

    print("-------------------")
    print("fit power spectra with power law")
    A_synch, alpha_synch, fit_min_s, fit_max = u.fit_routine(ell_arr, cls_s)
    A_dust, alpha_dust, fit_min_d, fit_max = u.fit_routine(ell_arr, cls_d)

    # update config file with fitted parameters
    for i, pols in enumerate(['TT', 'EE', 'BB']):
        config[f'A_synch_{pols}'] = A_synch[i]
        config[f'A_dust_{pols}'] = A_dust[i]
        config[f'alpha_synch_{pols}'] = alpha_synch[i]
        config[f'alpha_dust_{pols}'] = alpha_dust[i]

    # Copy the configuration file to output directory
    fit_paramfile = args.globals.replace(".yaml", "_fit.yaml")
    with open(fit_paramfile, "w") as f:
        yaml.dump(config, stream=f,
                  default_flow_style=False, sort_keys=False)

    if args.plots:
        print("-------------------")
        u.plotter_templates(plots_dir, maps_synch_dg, maps_dust_dg,
                            maps_synch, maps_dust,
                            mask, cls_s, cls_d,
                            A_synch, A_dust, alpha_synch, alpha_dust,
                            fit_min_s, fit_min_d, fit_max, ell_arr, ell_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract foregrounds spectral parameters in a sky region")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    templates_fit(args)
