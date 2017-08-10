# 3D-SDFGAN
3D Signed Distance Function Based Generative Adversarial Networks  
This study seeks to generate realistic looking, mesh-based 3D models by using GANs.

![Image of Car Sample](https://github.com/maxjiang93/SDFGAN/blob/combined/images/car_sample.png)

![Example 3D Model](https://github.com/skalnik/secret-bear-clip/blob/master/stl/clip.stl)
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

Borrows code from the following repositories:
 * DCGAN-tensorflow repository by Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
 * Pix2pix-tensorflow repository by affinelayer / [@affinelayer](https://github.com/affinelayer/pix2pix-tensorflow)
