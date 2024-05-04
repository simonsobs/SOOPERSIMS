import yaml
import argparse
import os
import numpy as np
import healpy as hp


def fcmb(nu):
    """
    Convert from thermodynamic (CMB) to Rayleigh-Jeans units,
    at frequency nu.
    -----------------
    input:
        nu: frequency
    output:
        RJ -> CMB conversion factor
    """
    x = 0.017608676067552197*nu
    ex = np.exp(x)
    return ex * (x / (ex-1))**2


def comp_sed(nu, nu0, beta, temp, typ):
    """
    SED of the components, returns thermodynamic (CMB) units.
    -----------------
    input:
        nu: frequency to scale to
        nu0: reference frequency for the component
        beta: SED power law index (e.g., beta_synch or beta_dust)
        typ: name of the component, can be 'cmb', 'dust' or 'synch'
    output:
        SED scaling factor, in thermodynamic (CMB) units
    """
    if typ == 'cmb':
        return fcmb(nu)
    elif typ == 'dust':
        x_to = 0.04799244662211351 * nu / temp
        x_from = 0.04799244662211351 * nu0 / temp
        ex_to = np.exp(x_to)
        ex_from = np.exp(x_from)
        return (nu/nu0)**(1+beta) * (ex_from-1)/(ex_to-1) * fcmb(nu0)/fcmb(nu)
    elif typ == 'synch':
        return (nu/nu0)**beta * fcmb(nu0) / fcmb(nu)
    return None


# def convolve_sed(nu_arr, bnu, sed):
#     """
#     """
#     dnu = np.zeros_like(nu_arr)
#     dnu[1:] = np.diff(nu_arr)
#     dnu[0] = dnu[1]
#     conv_sed = np.sum(dnu * bnu * nu_arr**2 * sed)
#     return conv_sed


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

    # create directories
    dirs = [out_dir, plots_dir]
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
    cl_synch[0] = A_s_TT * (ells / ell_0)**alpha_s_TT
    cl_synch[1] = A_s_EE * (ells / ell_0)**alpha_s_EE
    cl_synch[2] = A_s_BB * (ells / ell_0)**alpha_s_BB

    cl_dust[0] = A_d_TT * (ells / ell_0)**alpha_d_TT
    cl_dust[1] = A_d_EE * (ells / ell_0)**alpha_d_EE
    cl_dust[2] = A_d_BB * (ells / ell_0)**alpha_d_BB

    # extend TT at ell=0,1 to avoid inf
    cl_synch[0, :2] = cl_synch[0, 2]
    cl_dust[0, :2] = cl_dust[0, 2]
    # set ell=0,1 to 0 for polarization
    cl_synch[1:3, :2] = 0
    cl_dust[1:3, :2] = 0

    # SEDs
    sed_synch = {nu: comp_sed(nu, nu0_synch, beta_synch, None, 'synch')
                 for nu in freqs}
    sed_dust = {nu: comp_sed(nu, nu0_dust, beta_dust, T_dust, 'dust')
                for nu in freqs}

    # Generate alm drawing Gaussian random numbers from power spectra
    print("-------------------")
    print("Simulations")
    print("Generating alm")
    for seed in range(nsims):
        print("-------------------")
        print(f"- {seed:04d}")
        sims_dir = f"{out_dir}/sims/{seed:04d}/"
        os.makedirs(sims_dir, exist_ok=True)

        np.random.seed(seed)

        # generate alm from cl for all components
        alm_cmb = hp.synalm(cl_cmb, lmax=lmax, new=True)
        alm_synch = hp.synalm(cl_synch, lmax=lmax, new=True)
        alm_dust = hp.synalm(cl_dust, lmax=lmax, new=True)

        for nu in freqs:
            print(f"  {nu:03d} GHz")
            fname_out = f"{sims_dir}/alm_{nu:03d}GHz_lmax{lmax}_{seed:04d}.fits"  # noqa

            # TODO: bandpass integration (?)
            # bp_freqs, bp = bpass[nu]
            # conv_sed_synch = convolve_sed(bp_freqs, bp, sed_synch[nu])
            # conv_sed_dust = convolve_sed(bp_freqs, bp, sed_dust[nu])

            # rescale for component SEDs and coadd
            alm = alm_synch*sed_synch[nu] + alm_dust*sed_dust[nu] + alm_cmb
            # write to disk
            hp.write_alm(fname_out, alm, overwrite=True, out_dtype=np.float64)

    if args.plots:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        print("-------------------")
        print("plotting...")

        print("-------------------")
        print("plotting input C_ells")
        print("- CMB")
        hp_order = ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
        colors = ['k', 'r', 'b', 'y']
        for i, cl in enumerate(cl_cmb):
            plt.plot(cl, c=colors[i], label=hp_order[i])
        plt.loglog()
        plt.xlabel(r'multipole $\ell$')
        plt.ylabel(r'$C_\ell \, [\mu K^2]$')
        plt.title('CMB')
        plt.legend(frameon=False, ncols=2)
        plt.savefig(f"{plots_dir}/cl_cmb.png", bbox_inches='tight')
        plt.clf()

        print("- synchro")
        for i, cl in enumerate(cl_synch):
            plt.plot(cl, c=colors[i], label=hp_order[i])
        plt.axvline(ell_0, c='c', ls=':',
                    alpha=0.5, label=fr'$\ell={ell_0:d}$')
        plt.loglog()
        plt.xlabel(r'multipole $\ell$')
        plt.ylabel(r'$C_\ell \, [\mu K^2]$')
        plt.title('synchrotron')
        plt.legend(frameon=False)
        plt.savefig(f"{plots_dir}/cl_synch.png", bbox_inches='tight')
        plt.clf()

        print("- dust")
        for i, cl in enumerate(cl_dust):
            plt.plot(cl, c=colors[i], label=hp_order[i])
        plt.axvline(ell_0, c='c', ls=':',
                    alpha=0.5, label=fr'$\ell={ell_0:d}$')
        plt.loglog()
        plt.xlabel(r'multipole $\ell$')
        plt.ylabel(r'$C_\ell \, [\mu K^2]$')
        plt.title('dust')
        plt.legend(frameon=False)
        plt.savefig(f"{plots_dir}/cl_dust.png", bbox_inches='tight')
        plt.clf()

        print("-------------------")
        print("plotting SEDs")
        plt.plot(freqs, sed_synch.values(), '-o', label='synchro')
        plt.plot(freqs, sed_dust.values(), '-o', label='dust')
        plt.loglog()
        plt.xlabel('frequency [GHz]')
        plt.title('SEDs')
        plt.legend(frameon=False)
        plt.savefig(f"{plots_dir}/seds.png", bbox_inches='tight')
        plt.clf()

        print("-------------------")
        print("plotting an example of generated maps")
        print(f"seed {seed:04d} - {freqs[-1]:03d} GHz")
        m_cmb = hp.alm2map(alm_cmb, nside=nside)
        m_synch = hp.alm2map(alm_synch, nside=nside)
        m_dust = hp.alm2map(alm_dust, nside=nside)
        m_tot = hp.alm2map(alm, nside=nside)

        stokes = ['I', 'Q', 'U']
        units = r'$\mu K_{CMB}$'
        plt.figure(figsize=(14, 14))
        for i in range(3):
            hp.mollview(m_cmb[i], sub=(4, 3, (i+1)), cmap=cm.coolwarm,
                        unit=units, title=f'CMB {stokes[i]}')
            hp.mollview(m_synch[i], sub=(4, 3, (i+4)), cmap=cm.viridis,
                        unit=units, title=f'synch {stokes[i]}')
            hp.mollview(m_dust[i], sub=(4, 3, (i+7)), cmap=cm.inferno,
                        unit=units, title=f'dust {stokes[i]}')
            hp.mollview(m_tot[i], sub=(4, 3, (i+10)), cmap=cm.coolwarm,
                        unit=units, title=f'coadd {stokes[i]}')
        plt.savefig(f"{plots_dir}/sim_maps_{nu}GHz_{seed:04d}.png",
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alm sims for covariance computation")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    args = parser.parse_args()
    cov_sims(args)
