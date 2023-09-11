import os
import pandas as pd
import tarfile
import h5py
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pywt
from scipy.signal import welch

data_path = "Data"


for filename in os.listdir(data_path):
    if filename.endswith(".tar"):
        with tarfile.open(os.path.join(data_path, filename), 'r') as archive:
            archive.extractall(path=data_path)


for filename in os.listdir(data_path):
    if filename.endswith(".tar"):
        os.remove(os.path.join(data_path, filename))


s1_path = os.path.join(data_path, 's1')


def load_from_group(group):
    data = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            data[key] = load_from_group(item)
        elif isinstance(item, h5py.Dataset):
            data[key] = item[()]
    return data


with h5py.File(os.path.join(s1_path, 'data_primary.mat'), 'r') as file:
    data_primary = load_from_group(file)


summary_file_trial = pd.read_csv(os.path.join(s1_path, 'summary_file_trial.csv'), header=None)
summary_file_trial.columns = [
    "Trial number",
    "Accuracy",
    "Trial type",
    "Condition",
    "Delay jitter length in ms",
    "Response time in ms",
    "Pretrial epoch start time",
    "Encoding and pre-cue delay epoch start time",
    "Post-cue delay epoch start time"
]


neural_data = data_primary['gdat_clean_filt']


def segment_neural_data(neural_data, trial_data):
    segmented_trials = []
    for index, row in trial_data.iterrows():
        start_index = int(row['Pretrial epoch start time'])
        end_index = int(row['Post-cue delay epoch start time'] + row['Response time in ms'])
        trial_segment = neural_data[start_index:end_index, :]
        segmented_trials.append(trial_segment)
    return segmented_trials


segmented_trials = segment_neural_data(neural_data, summary_file_trial)


def extract_features(segmented_trials):
    means = []
    variances = []
    skews = []
    kurtoses = []
    for trial in segmented_trials:
        means.append(np.mean(trial, axis=0))
        variances.append(np.var(trial, axis=0))
        skews.append(skew(trial, axis=0))
        kurtoses.append(kurtosis(trial, axis=0))
    means = np.array(means)
    variances = np.array(variances)
    skews = np.array(skews)
    kurtoses = np.array(kurtoses)
    num_channels = means.shape[1]
    feature_names = []
    for i in range(num_channels):
        feature_names.extend([f"mean_ch{i}", f"variance_ch{i}", f"skew_ch{i}", f"kurtosis_ch{i}"])
    features_df = pd.DataFrame(np.hstack([means, variances, skews, kurtoses]), columns=feature_names)
    return features_df

X = extract_features(segmented_trials)  # Features
nan_columns = X.columns[X.isnull().all()].tolist()
X.drop(nan_columns, axis=1, inplace=True)
feature_names = list(X.columns)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = summary_file_trial['Trial type']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training

# Logistic Regression
clf = LogisticRegression(max_iter=5000)  # Initialize the classifier with more iterations
clf.fit(X_train, y_train)  # Train the classifier

# Model Evaluation for Logistic Regression
y_pred = clf.predict(X_test)  # Predict on test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred))

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Model Evaluation for Random Forest
rf_y_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("\nClassification Report for Random Forest:\n", classification_report(y_test, rf_y_pred))

# Extract feature importances from the trained Random Forest model
feature_importances = rf_clf.feature_importances_

# Combine feature names and their importance scores
features = list(zip(feature_names, feature_importances))

# Sort features based on importance
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

# Display top N features
N = 20
top_features = sorted_features[:N]

# Separate feature names and their scores for plotting
top_feature_names = [feature[0] for feature in top_features]
top_feature_scores = [feature[1] for feature in top_features]

# Plotting the feature importances
plt.figure(figsize=(15, 10))
plt.barh(top_feature_names, top_feature_scores, align='center')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top N Most Important Features from Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Optimizing the wavelet feature extraction:
def extract_wavelet_features(segmented_trials, max_length=5000):
    wavelet_features = []
    for trial in segmented_trials:
        coeffs = pywt.wavedec(trial, wavelet='db1', level=1)
        combined_coeffs = np.concatenate(coeffs)
        if len(combined_coeffs) > max_length:
            combined_coeffs = combined_coeffs[:max_length]
        elif len(combined_coeffs) < max_length:
            combined_coeffs = np.pad(combined_coeffs, (0, max_length - len(combined_coeffs)))
            
        wavelet_features.append(combined_coeffs)

    return np.array(wavelet_features)


def extract_psd_features(segmented_trials, fs=1000):  # Assuming a sampling rate of 1kHz
    psd_features_list = []

    # Determine the expected length of the power spectrum using the first trial
    frequencies, power_ref = welch(segmented_trials[0], fs=fs, nperseg=min(256, len(segmented_trials[0])))
    ref_length = power_ref.shape[1]
    
    for trial in segmented_trials:
        frequencies, power = welch(trial, fs=fs, nperseg=min(256, len(trial)))
        
        if power.shape[1] < ref_length:
            padding_length = ref_length - power.shape[1]
            padding_array = np.zeros((power.shape[0], padding_length))
            power = np.concatenate([power, padding_array], axis=1)
        else:
            power = power[:, :ref_length]

        flattened_power = power.flatten()
        psd_features_list.append(flattened_power)

    # Determine the desired length (using the max length here as an example)
    desired_length = max([item.shape[0] for item in psd_features_list])

    # Pad or truncate elements to the desired length
    for i in range(len(psd_features_list)):
        length_difference = desired_length - psd_features_list[i].shape[0]
        if length_difference > 0:  # If padding is needed
            psd_features_list[i] = np.concatenate([psd_features_list[i], np.zeros(length_difference)])
        else:  # If truncation is needed
            psd_features_list[i] = psd_features_list[i][:desired_length]

    return np.array(psd_features_list)


# Extract wavelet and PSD features
wavelet_features = extract_wavelet_features(segmented_trials)
psd_features = extract_psd_features(segmented_trials)

# Ensure wavelet_features and psd_features are 2D
if len(wavelet_features.shape) > 2:
    wavelet_features = wavelet_features.reshape(wavelet_features.shape[0], -1)

if len(psd_features.shape) > 2:
    psd_features = psd_features.reshape(psd_features.shape[0], -1)

# Combine the arrays
X_combined = np.hstack([X, wavelet_features, psd_features])

# Splitting the combined data into training and test sets (80% train, 20% test)
X_train_combined, X_test_combined, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Logistic Regression on Combined Features
clf_combined = LogisticRegression(max_iter=5000)
clf_combined.fit(X_train_combined, y_train)

# Model Evaluation for Logistic Regression on Combined Features
y_pred_combined = clf_combined.predict(X_test_combined)
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f"Logistic Regression (with combined features) Accuracy: {accuracy_combined * 100:.2f}%")
print("\nClassification Report for Logistic Regression with combined features:\n", classification_report(y_test, y_pred_combined))

# Random Forest on Combined Features
rf_clf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_combined.fit(X_train_combined, y_train)

# Model Evaluation for Random Forest on Combined Features
rf_y_pred_combined = rf_clf_combined.predict(X_test_combined)
rf_accuracy_combined = accuracy_score(y_test, rf_y_pred_combined)
print(f"\nRandom Forest (with combined features) Accuracy: {rf_accuracy_combined * 100:.2f}%")
print("\nClassification Report for Random Forest with combined features:\n", classification_report(y_test, rf_y_pred_combined))

# Extract feature importances from the trained Random Forest model
feature_importances_combined = rf_clf_combined.feature_importances_
features_combined = list(zip(feature_names + ["wavelet_" + str(i) for i in range(wavelet_features.shape[1])] + ["psd_" + str(i) for i in range(psd_features.shape[1])], feature_importances_combined))

# Sort features based on importance
sorted_features_combined = sorted(features_combined, key=lambda x: x[1], reverse=True)

# Display top N features
N = 20
top_features_combined = sorted_features_combined[:N]

# Separate feature names and their scores for plotting
top_feature_names_combined = [feature[0] for feature in top_features_combined]
top_feature_scores_combined = [feature[1] for feature in top_features_combined]

# Plotting the feature importances for combined features
plt.figure(figsize=(15, 10))
plt.barh(top_feature_names_combined, top_feature_scores_combined, align='center')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top N Most Important Features from Random Forest with Combined Features')
plt.gca().invert_yaxis()
plt.show()

