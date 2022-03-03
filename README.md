# Semi-HAR
This project is part of my master thesis at NTNU. It's a joint collaboration between SINTEF, NTNU and Hypersension. The object is to create a HAR algorithm that can support blood pressure measurments. The blood pressure measurment device (iNEMO) has no labeled data so therefore we we will create a semi-supervised learning algorithm from exisiting labeled datasets similar to the chest worn IMU that is on the iNEMO. The goal is to use the existing public labeled datasets as a base model and complement it with unlabeled data from the iNEMO.

## Install necessary modules with:
pip install -r requirements.txt

## Download and process raw dataset
To download and process the dataset run:

```
python3  run_data_generator.py --dataset all --mode download_and_process
```
This will download dataset [PAMAP2](http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) and [MHEALTH](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset) and commence raw processing of the datasets.
Information about the dataset is set in JSON file DATASET_META.

## Run Experiment
Experiment configuration is given inthe folder experiments and which file is chosen with --config (default='experiments/test_model.json').
Inside of the experiment configuration you can choose to use [WandB](https://wandb.ai/), which is an exellent tool to keep track of your experiment. To use this feature you have to create a user at their page.

Example of running a experiment:

```
python3  run_SSL.py --labelled_dataset MEALTH --unlabelled_dataset PAMAP2
```

run_SSL.py is for the moment not complete for the full semi-supervised experiment.

