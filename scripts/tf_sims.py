import yaml
import argparse
import os
import numpy as np
import healpy as hp
from datetime import date


def tf_sims(args):
    """
    Generate nsims simulations of spherical harmonics transform (alm)
    drawing Gaussian realizations from power-law power spectra.
    alm stored as separate files for pure T, pure E, and pure B cases:
    (alm_T, 0, 0)
    (0, alm_E, 0)
    (0, 0, alm_B)
    Final output is 3*nsims files.
    inputs:
        args: yaml configuration file with parameters
    outputs:
        simulated alms for transfer function estimation
    """

    # Load configuration file
    print("loading config file and parameters")
    fname_config = args.globals
    with open(fname_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = config['output_dir']
    plots_dir = out_dir + "/plots"
    config['date'] = date.today()

    dirs = [out_dir, plots_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Copy the configuration file to output directory
    with open(f"{out_dir}/config.yaml", "w") as f:
        yaml.dump(config, stream=f,
                  default_flow_style=False, sort_keys=False)

    # Get parameters from config file
    nside = config['nside']
    lmax = 3*nside - 1
    ells = np.arange(lmax+1)

    ell_0 = config['plaw_ell_pivot']
    A = config['plaw_amp']
    alpha = config['plaw_index']

    nsims = config['nsims']

    print("input parameters:")
    print(f" - nside = {nside} - lmax = {lmax}")
    print(f" - ell_pivot = {ell_0}")
    print(f" - amplitude = {A} - alpha = {alpha}")
    print(f" - number of simulations = {nsims}")

    hp_ordering = ['TT', 'TE', 'TB', 'EE', 'EB', 'BB']

    cl = np.zeros((6, len(ells)))
    # power-law spectra
    cl[:, :] = A * (ells / ell_0)**alpha
    # set ells=0,1 to 0
    cl[:, :2] = 0

    # Generate alm drawing Gaussian random numbers from power spectra
    print("-------------------")
    print("Simulating alm from power spectra")
    for seed in range(nsims):
        print(f"- {seed:04d}")
        sims_dir = f"{out_dir}/sims/{seed:04d}/"
        os.makedirs(sims_dir, exist_ok=True)

        fname_out_T = f"{sims_dir}/alm_tf_pureT_lmax{lmax}_{seed:04d}.fits.gz"  # noqa
        fname_out_E = f"{sims_dir}/alm_tf_pureE_lmax{lmax}_{seed:04d}.fits.gz"  # noqa
        fname_out_B = f"{sims_dir}/alm_tf_pureB_lmax{lmax}_{seed:04d}.fits.gz"  # noqa

        np.random.seed(seed)

        almT, almE, almB = hp.synalm(cl, lmax=lmax, new=False)

        alm_pureT = np.array([almT, 0*almE, 0*almB])
        alm_pureE = np.array([0*almT, almE, 0*almB])
        alm_pureB = np.array([0*almT, 0*almE, almB])

        hp.write_alm(fname_out_T, alm_pureT,
                     overwrite=True, out_dtype=np.float64)
        hp.write_alm(fname_out_E, alm_pureE,
                     overwrite=True, out_dtype=np.float64)
        hp.write_alm(fname_out_B, alm_pureB,
                     overwrite=True, out_dtype=np.float64)

    if args.plots:
        import matplotlib.pyplot as plt
        print("-------------------")
        print("plotting input power-law C_ells")
        for i, field_pair in enumerate(hp_ordering):
            plt.plot(cl[i], label=field_pair)
        plt.axvline(ell_0, c='k', ls=':', alpha=0.5, label=fr'$\ell={ell_0}$')
        plt.axhline(A, c='r', ls=':', alpha=0.5, label=fr'$A={A}$')
        plt.loglog()
        plt.ylim([1e-2, 1e6])
        plt.xlabel(r'multipole $\ell$')
        plt.ylabel(r'$C_\ell$')
        plt.title(fr"power law for transfer function: $\alpha={alpha}$")
        plt.legend(frameon=False, ncols=3)
        plt.savefig(f"{plots_dir}/plaw_tf.png", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for transfer function")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    tf_sims(args)
