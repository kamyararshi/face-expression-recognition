# face-expression-recognition
A Facial Expression Recognition Model trained on multiple dataset.
So far:

- *CKPlus* dataset [Link](https://www.jeffcohn.net/Resources/)
- *Emotic* dataet [Link](https://github.com/Tandon-A/emotic)


## Setup

create a conda environment by:

```bash
conda create -n faceexpr python=3.10
conda activate facexpr
```

and then instal requirements:

```bash
pip install -r requirements.txt
```

## Training

1. For the *CKPlus* dataset

- Create `dataset` folder:

```bash
mkdir dataset
```

- Unzip the `CK+.zip` dataset and put the folder `CK+` in the `dataset` folder
- Modify the config file `ckplus.yml`, placed in the `./configs/` directory, if needed
- training:
```bash
python train_ck.py
```

2. For the *Emotic* dataset

- Create `dataset` folder:

```bash
mkdir dataset
```

- Unzip the `emotic.zip` in the `dataset` folder, you will have a folder named `Emotic`, in which there will be `emotic` folder that has 4 folders inside
- Unzip the `Annotations.zip` in the `Emotic folder`, you will have a folder named `Annotations`
- You will have the following structure then:

```bash
├── ...
│   ├── emotic
│   |    ├── ade20k
│   |    ├── emodb_small
│   |    ├── framesdb
│   |    ├── mscoco 
│   ├── Annotations
│   |    ├── Annotations.mat
```

- To convert annotations from mat object to csv files and preprocess the data:
```bash
python ./codes/mat2py.py --data_dir ./dataset/Emotic/
```
**See this [repo](https://github.com/Tandon-A/emotic) for more info**

- Modify the config file `emotic.yml`, placed in the `./configs/` directory, if needed
- training:
```bash
python train_emotic.py
```

**All the logs and weights will be stored in `logs/` folder (look at the train scripts args for more detail)**

## Inference
to be added soon ...


## Note For Me:
**The motion_encoder.py in the codes folder contains the neural network. It should be modified and is not compatible yet to all datasets**