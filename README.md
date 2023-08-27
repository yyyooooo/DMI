# DMI
## Prerequisites
### Environment
- Python 3.6.13
- PyTorch 1.10.1
### Installation
Create a conda virtual environment and activate it
```
conda create -n DMI python=3.6.12
conda activate DMI
pip install -r requirements.txt
```
### Data Preparation
1. Dataset (SVD) is available here: [SVD](https://svdbase.github.io/)
2. Create a folder ${DATASET_DIR} to store the features. The directory structure should look like this:
```
${DATASET_DIR}
	|--${QUERY}
          |-- ${FEATURES}
          |-- ${INDEX}
          |-- ${SHAPE}
	|--${LABELED}
          |-- ${FEATURES}
          |-- ${INDEX}
          |-- ${SHAPE}
        |--${UNLABELED}
          |-- ${FEATURES}
          |-- ${INDEX}
          |-- ${SHAPE}
```
### Start Training
```
python main.py --cfg config/solver/DMI.yaml --log config/logging.yaml
```
### Start Test
```
python main.py --cfg config/solver/baseline.yaml --log config/logging.yaml --resume ${CHECKPOINT}
```
