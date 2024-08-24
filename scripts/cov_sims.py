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
    to be used for covariance estimation or transfer function validation.
    Inputs:
        args: yaml configuration file with parameters
    Outputs:
        simulated alms for covariance estimation / tf validation
    """

    # Load configuration file
    print("loading config file and parameters")
    fname_config = args.globals
    with open(fname_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = config['output_dir']
    plots_dir = out_dir + "/plots"
    cls_dir = out_dir + "/cls"
    config['output_units'] = 'K_CMB'
    config['date'] = date.today()

    # create directories
    dirs = [out_dir, plots_dir, cls_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Copy the configuration file to output directory
    with open(f"{out_dir}/config.yaml", "w") as f:
        yaml.dump(config, stream=f,
                  default_flow_style=False, sort_keys=False)

    # Get parameters from config file
    freqs = list(map(int, config['freqs']))
    nsims = config['nsims']
    nside = config['nside']
    lmax = 3*nside - 1
    ells = np.arange(lmax+1)

    # CMB
    cl_cmb = hp.read_cl(config['cmb_cls_path'])
    # synchrotron
    A_s_TT = config['A_synch_TT']
    A_s_EE = config['A_synch_EE']
    A_s_BB = config['A_synch_BB']
    alpha_s_TT = config['alpha_synch_TT']
    alpha_s_EE = config['alpha_synch_EE']
    alpha_s_BB = config['alpha_synch_BB']
    beta_synch = config['beta_synch']
    # dust
    A_d_TT = config['A_dust_TT']
    A_d_EE = config['A_dust_EE']
    A_d_BB = config['A_dust_BB']
    alpha_d_TT = config['alpha_dust_TT']
    alpha_d_EE = config['alpha_dust_EE']
    alpha_d_BB = config['alpha_dust_BB']
    beta_dust = config['beta_dust']
    T_dust = config['T_dust']

    nu0_synch = 23.
    nu0_dust = 353.
    ell_0 = int(config['plaw_ell_pivot'])

    print("config and input parameters for covariance sims:")
    print(f"input CMB: {config['cmb_cls_path']}")
    print(f"frequencies: {freqs}")
    print(f" - nside = {nside} -> lmax = {lmax}")
    print(f" - ell_pivot = {ell_0}")
    print(f" - number of simulations = {nsims}")

    # foregrounds C_ells from config (fitted) parameters
    # (TT, EE, BB, 0*TE)
    cl_synch = np.zeros((4, len(ells)))
    cl_dust = np.zeros((4, len(ells)))
    cl_synch[0] = u.plaw(ells, A_s_TT, alpha_s_TT)
    cl_synch[1] = u.plaw(ells, A_s_EE, alpha_s_EE)
    cl_synch[2] = u.plaw(ells, A_s_BB, alpha_s_BB)

    cl_dust[0] = u.plaw(ells, A_d_TT, alpha_d_TT)
    cl_dust[1] = u.plaw(ells, A_d_EE, alpha_d_EE)
    cl_dust[2] = u.plaw(ells, A_d_BB, alpha_d_BB)

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
    print("Simulations")
    print("Generating alm")
    for seed in mu.taskrange(nsims - 1):
        print("-------------------")
        print(f"- {seed:04d}")
        sims_dir = f"{out_dir}/sims/{seed:04d}/"
        os.makedirs(sims_dir, exist_ok=True)

        np.random.seed(seed)

        muK_to_K = 1.e-6

        # generate alm from cl for all components in K units
        alm_cmb = hp.synalm(cl_cmb, lmax=lmax, new=True) * muK_to_K
        alm_synch = hp.synalm(cl_synch, lmax=lmax, new=True) * muK_to_K
        alm_dust = hp.synalm(cl_dust, lmax=lmax, new=True) * muK_to_K

        for nu in freqs:
            print(f"  {nu:03d} GHz")
            fname_out = f"{sims_dir}/alm_{nu:03d}GHz_lmax{lmax}_{seed:04d}.fits"  # noqa

            # rescale for component SEDs and coadd
            alm = (alm_synch*sed_synch[nu] +
                   alm_dust*sed_dust[nu] +
                   alm_cmb*sed_cmb[nu])

            if config['bpass_integration']:
                print("bandpass integration not implemented yet")
                alm /= u.fcmb(nu)
            else:
                # back to CMB units
                alm /= u.fcmb(nu)

            # write alm to disk
            hp.write_alm(fname_out, alm, overwrite=True, out_dtype=np.float64)

    if args.plots:
        print("-------------------")
        print("plotting...")
        u.plotter_cov_sims(plots_dir, nside,
                           cl_cmb, cl_synch, cl_dust,
                           alm_cmb, alm_synch, alm_dust, alm,
                           seed, nu, ell_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for covariance computation")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    cov_sims(args)
