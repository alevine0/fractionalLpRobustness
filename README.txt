Code for the paper "Provable Adversarial Robustness for Fractional Lp Threat Models" by Alexander Levine and Soheil Feizi. Adapted from the code for Levine and Feizi (2021) available at https://github.com/alevine0/smoothingSplittingNoise; which is itself adapted from the code for Yang et al. (2020), available at https://github.com/tonyduan/rs4a.

Code is in the src directory.

Installation instructions:

"
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
"

To train using the L1 splitting method,

python3 -m src.train  --noise=SplitMethodDerandomized --seed=0 --s=${S}  --experiment-name=cifar_split_derandomized_${S} --output-dir checkpoints 

Note that the parameter `s' is equivalent to alpha in the paper.

To generate certificates using the L1 splitting method,

python3 -m src.test_derandomized --seed 0 --output-dir=checkpoints --noise=SplitMethodDerandomized --s ${S} --experiment-name=cifar_split_derandomized_${S} --noise-batch-size=128

To train using the L_1/2 splitting method,

python3 -m src.train  --noise=L_1_over_2_SplitMethodDerandomized --seed=0 --s=${S}  --experiment-name=cifar_l_half_derandomized_${S} --output-dir checkpoints 

To generate certificates using the L_1/2 splitting method,

python3 -m src.test_derandomized --seed 0 --output-dir=checkpoints --noise=L_1_over_2_SplitMethodDerandomized --s ${S} --experiment-name=cifar_l_half_derandomized_${S} --noise-batch-size=128


To train using the L_1/3 splitting method,

python3 -m src.train  --noise=L_1_over_3_SplitMethodDerandomized --seed=0 --s=${S}  --experiment-name=cifar_l_third_derandomized_${S} --output-dir checkpoints 

To generate certificates using the L_1/3 splitting method,

python3 -m src.test_derandomized --seed 0 --output-dir=checkpoints --noise=L_1_over_3_SplitMethodDerandomized --s ${S} --experiment-name=cifar_l_third_derandomized_${S} --noise-batch-size=128

There are also --noise  options for L_0 (L_0_SplitMethodDerandomized), Sparse (L_0_Pixel_SplitMethodDerandomized), L_1/2 with ``global lambda'' (GlobalLambda), and L_1/2 with arbitrary, rather than cyclic, permutations (L_1_over_2_SplitMethodDerandomizedPermute).

Note that for ImageNet training and testing, the directories where ImageNet files are expected are set using the bash variables $IMAGENET_TRAIN_DIR and $IMAGENET_TEST_DIR. 

******************
Additionally, the script ILP_for_L_lt_1.py solves an ILP to compute the distributions over Lambda for fractional Lp smoothing: these distributions have already been incorporated into the code. To attempt to find a distribution for p = 1/x, with error (epsilon in the paper) at most 0.02 and smoothing sample budget 1000, run:

python3 src.ILP_for_L_lt_1 --inverse_p x  --budget 1000  --approx_gap 0.02

Note that this script has the additional requirement of cvxpy.
