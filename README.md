# 3D-SDFGAN
3D Signed Distance Function Based Generative Adversarial Networks  
This study seeks to generate realistic looking, mesh-based 3D models by using GANs.

![Image of Car Sample](https://github.com/maxjiang93/SDFGAN/blob/combined/images/car_sample.png)

### Downloading the dataset
First, define the project directory to store the data and computed results (checkpoints, logs, samples). Change the project directory in the file `define_path.sh`:
```bash
# variables to define by user
PROJ_DIR=/Path/To/Your/Project/Directoy
JOB_NAME=job0-yourjobname
```
run the script to export variables to the current shell:
```
sudo chmod +x define_path.sh
./define_path.sh
```
go to project directory and download data:
```
cd $PROJ_DIR
wget _link_to_the_dataset_
tar -xvf data.tar
```

Borrows code from the following repositories:
 * DCGAN-tensorflow repository by Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
 * Pix2pix-tensorflow repository by affinelayer / [@affinelayer](https://github.com/affinelayer/pix2pix-tensorflow)
