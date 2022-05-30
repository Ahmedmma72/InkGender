import argparse
import os
import numpy as np
from utilityFunctions.utils import *
from utilityFunctions.hinge import get_hinge_features
from utilityFunctions.preprocess import *
from utilityFunctions.chainCodes import get_chain_code_features



parser = argparse.ArgumentParser(description='Extract features from images')
parser.add_argument('--input', type=str, default='./Training Data/')
parser.add_argument('--output', type=str, default='./Training Features/')
parser.add_argument('--model', type=str, default='./Model/')
args = parser.parse_args()

trainig_data_dir = args.input

# read training images
print("Reading training images...")
training_data, labels = load_images_from_folder(trainig_data_dir)

# preprocess training images

print("Preprocessing training images...")
for i in tqdm(range(len(training_data))):
    training_data[i] = preprocess(training_data[i])


# get features
print("Extracting hinge features...")
hinge_feature_vectors = []
for i in tqdm(range(len(training_data))):
    hinge_feature_vectors.append(get_hinge_features(training_data[i]))

print("Extracting chain code features...")
chain_code_feature_vectors = []
for i in tqdm(range(len(training_data))):
    chain_code_feature_vectors.append(
        get_chain_code_features(training_data[i])[0])

print("Combining features...")
features = []
for i in tqdm(range(len(training_data))):
    temp = np.append(hinge_feature_vectors[i], chain_code_feature_vectors[i])
    features.append(temp)



# save features
training_features_dir = args.output
if not os.path.exists(training_features_dir):
    os.makedirs(training_features_dir)


np.save(os.path.join(training_features_dir,
        f"features.npy"), features)
np.save(os.path.join(training_features_dir, f"labels.npy"), labels)

print("Features extracted and saved")