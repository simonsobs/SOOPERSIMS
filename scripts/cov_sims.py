import yaml
import argparse
import os
import numpy as np
import healpy as hp
from datetime import date
import utils as u
import mpi_utils as mu


def cov_sims(args):
    """
    Generate nsims simulations of spherical harmonics transform (alm)
    drawing Gaussian realizations from CMB and foregrounds power spectra,
    to be used for covariance estimation.
    Inputs:
        args: yaml configuration file with parameters
    Outputs:
        simulated alms for covariance estimation
    """

    # Load configuration file
    print("loading config file and parameters")
    fname_config = args.globals
    with open(fname_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = config['output_dir']
    plots_dir = f"{out_dir}/plots"
    cls_dir = f"{out_dir}/cls"
    config['output_units'] = 'uK_CMB'
    config['date'] = date.today()

    # create directories
    dirs = [out_dir, plots_dir, cls_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    fgs_model = config['fgs_model']
    read_alm = config['read_alm']
    store_alm = config['store_alm']
    store_maps = config['store_maps']
    if store_maps:
        pix_type = config['pix_type']
        if pix_type == 'car':
            from pixell import enplot, enmap, curvedsky
            fname_car_template = "/pscratch/sd/c/cranucci/BB/SOOPERSIMS/data/band_car_fejer1_5arcmin.fits"
            fname_car_template = config['car_template']
            shape, wcs = enmap.read_map_geometry(fname_car_template)
            res = int(config['res'])

    # Copy the configuration file to output directory
    fname_config_out = f"{out_dir}/config_{fgs_model}"
    if store_maps:
        fname_config_out += f"_{pix_type}"
        if pix_type == 'car':
            fname_config_out += f"_{res}arcmin"
    fname_config_out += ".yaml"
    with open(fname_config_out, "w") as f:
        yaml.dump(config, stream=f,
                  default_flow_style=False, sort_keys=False)

    # Get parameters from config file
    freqs = list(map(int, config['freqs']))
    nfreqs = len(freqs)
    nsims = config['nsims']
    nside = config['nside']
    lmax = 3*nside - 1
    ells = np.arange(lmax + 1)

    smooth = config['SAT_beams']
    if smooth:
        fwhm_amin = {'SATp1_f090': 28.4, 'SATp1_f150': 20.0,
                     'SATp3_f090': 28.0, 'SATp3_f150': 19.4}
        fwhm_rad = {key: np.deg2rad(fwhm_amin[key] / 60.)
                    for key in fwhm_amin.keys()}
        nbeams = int(len(fwhm_amin.keys()) / 2)
        if nfreqs != nbeams:
            print("ERROR: number of frequencies and number of beams must match.")
            print(f"nfreqs: {nfreqs} != nbeams: {nbeams}")
            import sys
            sys.exit(0)

    # CMB
    cl_cmb = hp.read_cl(config['cmb_cls_path'])
    # foregrounds
    beta_synch = config['beta_synch']
    beta_dust = config['beta_dust']
    T_dust = config['T_dust']
    A_s, A_d = {}, {}
    alpha_s, alpha_d = {}, {}
    for pols in ['TT', 'EE', 'BB']:
        A_s[pols] = config[f'A_synch_{pols}']
        alpha_s[pols] = config[f'alpha_synch_{pols}']
        A_d[pols] = config[f'A_dust_{pols}']
        alpha_d[pols] = config[f'alpha_dust_{pols}']

    nu0_synch = 23.
    nu0_dust = 353.
    ell_0 = int(config['plaw_ell_pivot'])

    print("config and input parameters for covariance sims:")
    print(f"input CMB: {config['cmb_cls_path']}")
    print(f"frequencies: {freqs}")
    print(f" - nside = {nside} -> lmax = {lmax}")
    print(f" - ell_pivot = {ell_0}")
    print(f" - number of simulations = {nsims}")

    print("-------------------")
    print("building foregrounds C_ells from fitted parameters")
    # (TT, EE, BB, 0*TE)
    cl_synch = np.zeros((4, len(ells)))
    cl_dust = np.zeros((4, len(ells)))
    for i, pols in enumerate(['TT', 'EE', 'BB']):
        cl_synch[i] = u.plaw(ells, A_s[pols], alpha_s[pols])
        cl_dust[i] = u.plaw(ells, A_d[pols], alpha_d[pols])

    # extend TT at ell=0,1 to avoid inf
    cl_synch[0, :2] = cl_synch[0, 2]
    cl_dust[0, :2] = cl_dust[0, 2]
    # set ell=0,1 to 0 for polarization
    cl_synch[1:3, :2] = 0
    cl_dust[1:3, :2] = 0

    # SEDs
    sed_synch = {nu: u.comp_sed(nu, nu0_synch, beta_synch, None, 'synch')
                 for nu in freqs}
    sed_dust = {nu: u.comp_sed(nu, nu0_dust, beta_dust, T_dust, 'dust')
                for nu in freqs}
    sed_cmb = {nu: u.comp_sed(nu, None, None, None, 'cmb')
               for nu in freqs}

    # Save theory power spectra
    for inu1, nu1 in enumerate(freqs):
        for inu2, nu2 in enumerate(freqs):
            if inu2 < inu1:
                continue
            cl_dust_scaled = cl_dust * (sed_dust[nu1] / u.fcmb(nu1)
                                        * sed_dust[nu2] / u.fcmb(nu2))
            cl_synch_scaled = cl_synch * (sed_synch[nu1] / u.fcmb(nu1)
                                          * sed_synch[nu2] / u.fcmb(nu2))
            hp.write_cl(
                f"{cls_dir}/cl_dust_f{str(nu1).zfill(3)}_f{str(nu2).zfill(3)}.fits",
                cl_dust_scaled, overwrite=True
            )
            hp.write_cl(
                f"{cls_dir}/cl_synch_f{str(nu1).zfill(3)}_f{str(nu2).zfill(3)}.fits",
                cl_synch_scaled, overwrite=True
            )
            hp.write_cl(
                f"{cls_dir}/cl_cmb.fits",
                cl_cmb, overwrite=True
            )

    # Initialize MPI
    mu.init(config['mpi_bool'])

    # Generate alm drawing Gaussian random numbers from power spectra
    print("-------------------")
    print("Generating simulations")
    for seed in mu.taskrange(nsims - 1):
        print(f"{seed:04d}")
        sims_dir = f"{out_dir}/sims/{seed:04d}"
        if not read_alm:
            os.makedirs(sims_dir, exist_ok=True)

            # generate alm from cl for all components in muK units
            np.random.seed(seed)
            alm_cmb = hp.synalm(cl_cmb, lmax=lmax, new=True)
            alm_synch = hp.synalm(cl_synch, lmax=lmax, new=True)
            alm_dust = hp.synalm(cl_dust, lmax=lmax, new=True)

        for nu in freqs:
            if store_maps and read_alm:
                if smooth:
                    fnames_alm = [f"{sims_dir}/alm_{fgs_model}_{nu:03d}GHz_SATp1_lmax{lmax}_{seed:04d}.fits",  # noqa
                                  f"{sims_dir}/alm_{fgs_model}_{nu:03d}GHz_SATp3_lmax{lmax}_{seed:04d}.fits"]  # noqa
                    alm_satp1 = hp.read_alm(fnames_alm[0], hdu=(1, 2, 3))
                    alm_satp3 = hp.read_alm(fnames_alm[1], hdu=(1, 2, 3))
                else:
                    fname_alm = f"{sims_dir}/alm_{fgs_model}_{nu:03d}GHz_lmax{lmax}_{seed:04d}.fits"  # noqa
                    alm = hp.read_alm(fname_alm, hdu=(1, 2, 3))
            else:
                # rescale for component SEDs and coadd
                alm = (alm_synch * sed_synch[nu] +
                       alm_dust * sed_dust[nu] +
                       alm_cmb * sed_cmb[nu])
    
                if config['bpass_integration']:
                    print("bandpass integration not implemented yet")
                    alm /= u.fcmb(nu)
                else:
                    # back to CMB units
                    alm /= u.fcmb(nu)

                if smooth:
                    # smooth alm with SATs beams
                    alm_satp1 = hp.smoothalm(alm,
                                             fwhm=fwhm_rad[f'SATp1_f{nu:03d}'],
                                             inplace=False)
                    alm_satp3 = hp.smoothalm(alm,
                                             fwhm=fwhm_rad[f'SATp3_f{nu:03d}'],
                                             inplace=False)

                if store_alm:
                    if smooth:
                        # write alm to disk (smoothed)
                        for alm_beam, sat in zip([alm_satp1, alm_satp3], ['SATp1', 'SATp3']):
                            fname_out = f"{sims_dir}/alm_{fgs_model}_{nu:03d}GHz_{sat}_lmax{lmax}_{seed:04d}.fits"  # noqa
                            hp.write_alm(fname_out, alm_beam,
                                         overwrite=True, out_dtype=np.float64)
                    else:
                        # write alm to disk (no beam smoothing)
                        fname_out = f"{sims_dir}/alm_{fgs_model}_{nu:03d}GHz_lmax{lmax}_{seed:04d}.fits"  # noqa
                        hp.write_alm(fname_out, alm, overwrite=True, out_dtype=np.float64)

            if store_maps:
                # project alm to maps and write to disk
                if pix_type == 'hp':
                    if smooth:
                        for alm_beam, sat in zip([alm_satp1, alm_satp3], ['SATp1', 'SATp3']):
                            maps_beam = hp.alm2map(alm_beam, nside=nside)
                            fname_out = f"{sims_dir}/maps_hp_{fgs_model}_{nu:03d}GHz_{sat}_nside{nside}_{seed:04d}.fits"  # noqa
                            hp.write_map(fname_out, maps_beam, dtype=np.float64, overwrite=True)
                    else:
                        maps = hp.alm2map(alm, nside=nside)
                        fname_out = f"{sims_dir}/maps_hp_{fgs_model}_{nu:03d}GHz_nside{nside}_{seed:04d}.fits"  # noqa
                        hp.write_map(fname_out, maps, dtype=np.float64, overwrite=True)

                elif pix_type == 'car':
                    if smooth:
                        for alm_beam, sat in zip([alm_satp1, alm_satp3], ['SATp1', 'SATp3']):
                            maps_beam = enmap.zeros((3,) + shape, wcs)
                            curvedsky.alm2map(alm_beam, maps_beam)
                            fname_out = f"{sims_dir}/maps_car_{fgs_model}_{nu:03d}GHz_{sat}_{res}arcmin_{seed:04d}.fits"  # noqa
                            enmap.write_map(fname_out, maps_beam)
                    else:
                        maps = enmap.zeros((3,) + shape, wcs)
                        curvedsky.alm2map(alm, maps)
                        fname_out = f"{sims_dir}/maps_car_{fgs_model}_{nu:03d}GHz_{res}arcmin_{seed:04d}.fits"  # noqa
                        enmap.write_map(fname_out, maps)

    if args.plots:
        print("-------------------")
        print("plotting")
        if pix_type == 'hp':
            if smooth:
                alm = alm_beam
            if not read_alm:
                u.plotter_cov_sims(plots_dir, nside,
                                   cl_cmb, cl_synch, cl_dust,
                                   alm_cmb, alm_synch, alm_dust, alm,
                                   seed, nu, ell_0)
            else:
                import matplotlib.pyplot as plt
                from matplotlib import cm
                units = r'$\mu \mathrm{K_{CMB}}$'
                if smooth:
                    maps = maps_beam
                    fname_out = f"{plots_dir}/sim_maps_hp_{nu:03d}GHz_{sat}_nside{nside}_{seed:04d}.png"
                else:
                    fname_out = f"{plots_dir}/sim_maps_hp_{nu:03d}GHz_nside{nside}_{seed:04d}.png"
                hp.projview(maps[1], cmap=cm.coolwarm, unit=units,
                            llabel=f"{nu} GHz", rlabel="Q")
                plt.savefig(fname_out, bbox_inches='tight')
                plt.close()
        elif pix_type == 'car':
            if store_maps:
                vrange = 7.5
                s = 1
                if smooth:
                    plot = enplot.plot(maps_beam[s], downgrade=4 if res<5 else 2,
                                       colorbar=False, min=-vrange, max=vrange)
                    enplot.write(f"{plots_dir}/sim_maps_car_{nu:03d}GHz_{sat}_{res}arcmin_{seed:04d}", plot)
                else:
                    plot = enplot.plot(maps[s], downgrade=4 if res<5 else 2,
                                       colorbar=False, min=-vrange, max=vrange)
                    enplot.write(f"{plots_dir}/sim_maps_car_{nu:03d}GHz_{res}arcmin_{seed:04d}", plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for covariance computation")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    cov_sims(args)
