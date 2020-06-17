# Classification Under Human Assistance

This is a repository containing the code and data for the paper:

> A. De, N. Okati, A. Zarezade and M. Gomez-Rodriguez. _Classification Under Human Assistance._

The paper is available [on archive](TBD).

## Pre-requisites

This code depends on the following packages:

 1. `numpy`
 2. `scipy`
 3. `matplotlib`
 4. `sklearn`
 5. `Keras (for preprocessing image datasets)`
 6. `Tensorflow (for preprocessing image datasets)`
 

## Code structure

 - `algorithms.py and baseline_classes.py` contain the implementation of our algorithms and the baselines.
 - `train.py` training script.
 - `test.py` testing and plotting script
 - `generate_human_error.py` generates human error for the image datasets.
 - `generate_synthetic.py` generates synthetic datasets.
 - `preprocess.py` preprocessing and feature extraction of image datasets script
 - `cross_validate.py` finds the best \lambda value using cross validation
 - `/Results` contains pretrained models
 - `/plots` contains the figures of the paper
 - `/data` preprocessed real datasets and generated synthetic datasets are saved in this folder.


## Execution

####Test and generate the figures:
`python test.py --dataset=dataset_name --svm_type=soft_linear_with_offset`

For example:
`python test.py --dataset=Messidor --svm_type=soft_linear_with_offset`

The figures will be saved in `/plots` folder.


####Run the algorithms:
`python train.py --dataset=dataset_name --svm_type=soft_linear_with_offset`

For example:
`python train.py --dataset=Stare --svm_type=soft_linear_with_offset`

Available datasets : `['Messidor', 'Stare', 'Aptos', 'Linear','Kernel']`

- Messidor, Stare and Aptos are the real datasets.

- Linear and Kernel are the synthetic datasets.

All the default parameters are set based on the paper. You can change them in the code.

The results will be saved in `/Results` folder. The results corresponding to all datasets are already generated and saved in `/Results`.

####Regenerate synthetic data:
**There is no need to run this script if you want to use the same synthetic data as mentioned in the paper**, but, if you wish to generate synthetic datasets with new settings you can modify the `generate_synthetic.py` script and then run:

`python generate_synthetic.py name_of_the_synthetic_dataset`

For example:
`python generate_synthetic.py Linear`

Synthetic datasets : `['Linear','Kernel']`

and then train and test to see the new results.

####Change the human error for the real datasets:

`python generate_human_error.py name_of_image_dataset`

For example:
`python generate_human_error.py Stare`

image datasets are : `['Messidor,'Stare','Aptos']`

and then train and test to see the new results.

----

## Pre-processing

The datasets are preprocessed and saved in `data` folder and **there is no need to download them again**, but, if you wish to change the preprocessing or feature extraction method, you may download the [Messidor](http://www.adcis.net/en/third-party/messidor/), [Stare](https://cecas.clemson.edu/~ahoover/stare/), and [Aptos](https://www.kaggle.com/c/aptos2019-blindness-detection/overview/aptos-2019) datasets and use `preprocess.py` to preprocess them. You will also need [Resnet](https://github.com/KaimingHe/deep-residual-networks) or [VGG16](https://neurohive.io/en/popular-networks/vgg16/) to generate feature vectors of the image datasets.

`python preprocess.py`

## Cross Validation
We found \lambda values using cross validation. The code can be run using:

`python cross_validate.py dataset_name`

For example:
`python cross_validate.py Stare`

## Plots
Plots are generated using `python test.py --dataset=dataset_name --svm_type=svm_type`

All plots are generated and saved in `/plots` folder.
