import argparse
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


parser = argparse.ArgumentParser(description='training')
parser.add_argument('--input', type=str, default='./Training Features/')
parser.add_argument('--output', type=str, default='./Model/')

args = parser.parse_args()

training_features_dir = args.input

# load features
print("Loading features...")
features = np.load(os.path.join(
    training_features_dir, f"features.npy"))
labels = np.load(os.path.join(training_features_dir, f"labels.npy"))

# RandomForestClassifier
clf = RandomForestClassifier(n_estimators=210, random_state=0)

print("Training RandomForestClassifier...")
clf.fit(features, labels)

# save model

model_dir = args.output


pickle.dump(clf, open(os.path.join(
    model_dir, "RandomForestClassifier.pkl"), 'wb'))


print('model trained and saved')
