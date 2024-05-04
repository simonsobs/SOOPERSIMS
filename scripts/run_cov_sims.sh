paramfile='../paramfiles/paramfile_cov.yaml'
paramfile_fit='../paramfiles/paramfile_cov_fit.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "-------------------"
echo "estimating foregrounds spectral parameters from templates..."
echo "-------------------"
python templates_fit.py --globals ${paramfile} --plots

echo "-------------------"
echo "generating covariance simulations..."
echo "-------------------"
python cov_sims.py --globals ${paramfile_fit} --plots
