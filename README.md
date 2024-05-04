# SOOPERSIMS
Library to generate simulations for the `SOOPERCOOL` pipeline, for transfer function or covariance estimation.

## Requirements
- `numpy`
- `scipy`
- `healpy` (https://healpy.readthedocs.io/en/latest/)
- `pymaster` (https://namaster.readthedocs.io/en/latest/)

## Installation
Just clone or download the repository, for example with:
`git clone https://github.com/simonsobs/SOOPERSIMS.git`

## Run
Scripts need a `.yaml` configuration file with instructions and parameters. Create a `.yaml` file in the `SOOPERSIMS/paramfiles` directory (sample files there)

To run, go to the `scripts` directory (`cd SOOPERSIMS/scripts`) and run the bash scripts:
- `bash run_tf_sims.sh`
to generate transfer function simulations
- `bash run_cov_sims.sh`
to generate covariance simulations

## Contacts
Get in touch with Claudio Ranucci (cranucci) if you have questions or feedbacks about the codes.
