import argparse
import os
import numpy as np
from utilityFunctions.utils import *
from utilityFunctions.hinge import get_hinge_features
from utilityFunctions.preprocess import *
from sklearn.svm import SVC
from utilityFunctions.chainCodes import get_chain_code_features
import pickle
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--test', type=str, default='./Testing Data/')
parser.add_argument('--output', type=str, default='./Testing Results/')
parser.add_argument('--model', type=str, default='./Model/')
args = parser.parse_args()

# load model 
clf = pickle.load(
    open(os.path.join(args.model, "RandomForestClassifier.pkl"), 'rb'))


# load testing images
test_data_dir = args.test
print("Reading testing images...")
test_data, _ = load_images_from_folder(test_data_dir)


predictions = []
times = []

# start the pipeline

print("Predicting...")

for i in tqdm(range(len(test_data))):
    start = time.time()                                   # start timer
    test_data[i] = preprocess(test_data[i])               # preprocess image
    hinge = get_hinge_features(test_data[i])              # get hinge features
    chainCodes = get_chain_code_features(test_data[i])[0] # get chain code features
    feature = np.append(hinge, chainCodes)                # combine features
    feature = feature.reshape(1, -1)
    prediction = clf.predict(feature)                     # predict class
    end = time.time()                                     # end timer
    total_time = end - start                              # calculate total time
    if total_time == 0:
        total_time = 0.001                                # 1 ms tollerance
    times.append(total_time)
    predictions.append(prediction)


# save results
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


with open(os.path.join(output_dir, "results.txt"), 'w') as f:
    for i in range(len(predictions)):
        f.write(str(predictions[i][0]) + '\n')

# save times to file times.txt in output directory
with open(os.path.join(output_dir, "times.txt"), 'w') as f:
    for i in range(len(times)):
        f.write(str(times[i]) + '\n')


print("Done predicting ready for evaluation.......................")
