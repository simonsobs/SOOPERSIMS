# config files with parameters
paramfile='../paramfiles/paramfile_tf.yaml'

echo "Running scripts with paramfile: ${paramfile}"

echo "-------------------"
echo "transfer function simulations..."
echo "-------------------"
python tf_sims.py --globals ${paramfile} --plots
