# Transfer-HAR
This project is part of my master thesis at NTNU. It's a joint collaboration between SINTEF, NTNU and Hypersension. The object is to create a HAR algorithm that can support blood pressure measurments. The blood pressure measurment device (iNEMO) has no labeled data so therefore we we will create a transfer learning algorithm from exisiting labeled datasets similar to the chest worn IMU that is on the ISenseU device. The goal is to use the existing public labeled datasets as a base model and complement it with unlabeled data from the iNEMO.
This github is not complete. The data generator script and processing of the raw data is complete. For the Transfer Learning approah see my Kaggle: https://www.kaggle.com/mnotoe/trans-har/edit


## Install necessary modules with:
pip install -r requirements.txt

