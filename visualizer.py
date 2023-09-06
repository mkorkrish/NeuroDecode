import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
summary_file_trial_df = pd.read_csv('Files\Data\s1\summary_file_trial.csv')
summary_file_trial_df.columns = [
    "Trial number", "Accuracy", "Trial type", "Condition", 
    "Delay jitter length in ms", "Response time in ms", 
    "Pretrial epoch start time", "Encoding and pre-cue delay epoch start time", 
    "Post-cue delay epoch start time"
    ]

# Visualizations

# Setting up the visual settings
plt.figure(figsize=(20, 10))

# Accuracy by Trial Type
plt.subplot(2, 3, 1)
sns.barplot(data=summary_file_trial_df, x="Trial type", y="Accuracy")
plt.title("Accuracy by Trial Type")
plt.xticks(ticks=[0, 1], labels=["Identity", "Relation"])
plt.ylabel("Average Accuracy")
plt.xlabel("Trial Type")

# Response Time Distribution
plt.subplot(2, 3, 2)
sns.histplot(data=summary_file_trial_df, x="Response time in ms", bins=30, kde=True)
plt.title("Distribution of Response Times")
plt.xlabel("Response Time (ms)")
plt.ylabel("Count")

# Relation Between Response Time and Accuracy
plt.subplot(2, 3, 3)
sns.boxplot(data=summary_file_trial_df, x="Accuracy", y="Response time in ms")
plt.title("Relation Between Response Time and Accuracy")
plt.xticks(ticks=[0, 1], labels=["Incorrect", "Correct"])
plt.xlabel("Accuracy")
plt.ylabel("Response Time (ms)")

# Evolution of Response Time over Trials
plt.subplot(2, 3, 4)
sns.lineplot(data=summary_file_trial_df, x="Trial number", y="Response time in ms")
plt.title("Evolution of Response Time over Trials")
plt.xlabel("Trial Number")
plt.ylabel("Response Time (ms)")

# Distribution of Response Times by Condition
plt.subplot(2, 3, 5)
sns.violinplot(data=summary_file_trial_df, x="Condition", y="Response time in ms")
plt.title("Distribution of Response Times by Condition")
plt.xticks(ticks=[0, 1, 2], labels=["Identity", "Spatial Relation", "Temporal Relation"])
plt.xlabel("Condition")
plt.ylabel("Response Time (ms)")

plt.tight_layout()
plt.show()
