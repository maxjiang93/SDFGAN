# 3D-SDFGAN
3D Signed Distance Function Based Generative Adversarial Networks  

![Image of Car Sample](https://github.com/maxjiang93/SDFGAN/blob/master/images/car_15_small.gif)

![3D Model of Above Car Sample](https://github.com/maxjiang93/SDFGAN/blob/master/images/car_sample.stl) 

![Rendered Furniture Samples (Chairs and Tables)](https://github.com/maxjiang93/SDFGAN/blob/master/images/furniture-render.gif)

## About this study
This study seeks to generate realistic looking, mesh-based 3D models by using GANs. Training is based on [ShapeNetCore dataset](https://www.shapenet.org/) that has been post-processed into 64x64x64 signed distance function fields. More details about this study can be found in [this paper](https://arxiv.org/abs/1709.07581)

## Reproducing the results
### Collecting the dataset
First, define the project directory to store the data and computed results (checkpoints, logs, samples). Change the project directory, job name and dataset to train in the file `define_path.sh`. These parameters will be called globally throughout the training and testing:
```bash
# variables to define by user
PROJ_DIR=/Path/To/Your/Project/Directoy
JOB_NAME=job0-yourjobname
DATASET=synset_02958343_car
```
run the script to export variables to the current shell:
```
sudo chmod +x *.sh
./define_path.sh
```
go to project directory and download data:
```
cd $PROJ_DIR
wget _link_to_the_dataset_
tar -xvf data.tar
```
go back to code directory to run frequency spliting code for a dataset (e.g. car dataset) for running pix2pix:
```
cd /Path/To/SDFGAN
python stream_freqsplit.py ${PROJ_DIR}/data/synset_02958343_car ${PROJ_DIR}/data/synset_02958343_car_freqsplit 0 0
```
the above code will run the freqsplit algorithm on multiple threads. Trailing 0 and 0 defaults the code to use all available threads and convert all data within the dataaset.

### Running the sdfgan part of the code
```
./train_sdfgan.sh
```
### Running the pix2pix part of the code
```
./train_pix2pix.sh
```
### Generate test results
```
./test_all.sh
```
### Post processing for mesh
Codes for postprocessing the results to create mesh is not including in this current repository. However, it can be easily achieved using a Marching-Cubes Algorithm. The mesh in the example above is further post-processed using 3 steps of Laplacian smoothing, quadratic mesh decimation by half, and hole-filling for the holes at the front and back as a result of boundary effects caused by the limitations of the bounding box.

## Open Source Code Credits
Borrows code from the following repositories:
 * [DCGAN-tensorflow](http://carpedm20.github.io/) repository by Taehoon Kim
 * [Pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) repository by affinelayer

Relied on the following Open Source projects for 3D pre and post processing of 3D mesh / signed distance function:
 * [Libigl](https://github.com/libigl/libigl) interactive graphics library by Alec Jacobson et. al.
 * [Meshlab](http://www.meshlab.net/) for mesh rendering and minor post-processing.
