# SOOPERSIMS
Library to generate simulations for the `SOOPERCOOL` pipeline, for transfer function estimation or covariance matrix calculation.

---

## Codes description
These scripts generate two type of simulations:
- simulations for transfer function estimation
- simulations for covariance matrix calculation

### Transfer function type
The transfer function can be estimated by computing the power spectra of simulated maps
before filtering and comparing them with those of the same maps after filtering. <br />
The `tf_sims.py` script produces the spherical harmonics coefficients ($a_{\ell m}$) of these maps, up to $\ell_{max}=3*nside-1$, where `nside` can be configured.
The code generates the $a_{\ell m}$ by drawing Gaussian random numbers from a simple input power spectrum, of the form $$C_\ell = A \left(\frac{\ell}{\ell_0}\right)^\alpha.$$
Amplitude ($A$) and slope ($\alpha$) can be configured; values of $\alpha \in [-3,0]$ cover the typical steepness of Gaussian and CMB-like spectra. <br />
Spherical harmonics coefficients are saved to disk in $\mu K_{CMB}$ units as three separated files, in `.fits.gz` format:

- T-mode only ($a^T_{\ell m}$, $\quad$ 0, $\quad$ 0) `alm_tf_pureT_lmax{lmax}_{seed}.fits.gz`
- E-mode only (0, $\quad$ $a^E_{\ell m}$, $\quad$ 0) `alm_tf_pureE_...`
- B-mode only (0, $\quad$ 0, $\quad$ $a^B_{\ell m}$) `alm_tf_pureB_...` <br />

### Covariance type
Generation of simulations for covariance estimation is separated in two scripts:

- `templates_fit.py` first reads the PySM `d9s4` foregrounds templates for synchrotron ($\nu_0=23$ GHz) and thermal dust ($\nu_0=353$ GHz) stored at NERSC. IQU maps are converted from Rayleigh-Jeans ($\mu K_{RJ}$) to thermodynamic (CMB, $\mu K_{CMB}$) units and processed into a SAT-like configuration, meaning that they are rotated to equatorial coordinates and downgraded to the desired `nside` (both operations are carried out in harmonic space). Then, for each template, the script computes power spectra with `NaMaster` on the SAT MSS2 apodized mask and extracts power laws amplitudes and slopes by fitting the spectra with: $$C_\ell = A_c \left(\frac{\ell}{\ell_0}\right)^{\alpha_c}$$ where $c=s, d$ stands for synchrotron or dust. Finally, it updates the `paramfile_cov.yaml` file with the fitted parameters for it to be read by the simulation script.

- `cov_sims.py` generates Gaussian $a_{\ell m}$ realizations from the power spectra of each component (CMB, synchrotron and dust), up to $\ell_{max}=3*nside-1$. The code loads CMB input $C_\ell$s from the `/data` directory (default is *Planck* 2018), while foregrounds spectra are computed as power laws $C_\ell = A_c \left(\ell/\ell_0\right)^{\alpha_c}$ with spectral parameters fitted in the previous stage. In order to generate multi-frequency realizations, each $a_{\ell m}$ is frequency-scaled with the corresponding spectral energy distribution (SED). Synchrotron radiation is rescaled with a power law ($\beta_s = -3$); thermal dust is rescaled with a modified blackbody ($\beta_d=1.59$, $T_d = 19.6$). Bandpass integration is <u>not</u> included; frequency bandpasses are assumed as delta-like, centered at the nominal frequencies. Outputs spherical harmonics coefficients are saved to disk in $\mu K_{CMB}$ units, the filename being `alm_{freq}GHz_lmax{lmax}_{seed}.fits`.

---

## Requirements
- `numpy`
- `scipy`
- `healpy` (https://healpy.readthedocs.io/en/latest/)
- `pymaster` (https://namaster.readthedocs.io/en/latest/)

## Installation
Just clone or download the repository, for example with:
`git clone https://github.com/simonsobs/SOOPERSIMS.git`

## Run
Scripts need a `.yaml` configuration file with instructions and parameters. Create a `.yaml` file in the `SOOPERSIMS/paramfiles` directory (or just adapt on of the sample files there to your needs). <br />
To run, go to the `scripts` directory (`cd SOOPERSIMS/scripts`) and run the bash scripts:
- `bash run_tf_sims.sh` to generate transfer function simulations
- `bash run_cov_sims.sh` to generate covariance simulations

---

## Contacts
Get in touch with Claudio Ranucci (cranucci) if you have questions or feedbacks about the codes.
