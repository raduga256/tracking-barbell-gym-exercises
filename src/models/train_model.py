import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

# Fixing import failures
import sys
sys.path.append('c:/Users/memor/Desktop/Projects-to-Complete/ml-fitness-tracker/src/models')
from LearningAlgorithms import ClassificationAlgorithms


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Read the data into a pandas DataFrame

df =pd.read_pickle(r"..\..\data\interim\03_data_features.pkl")
df.head()
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# drop some unnecessary columns
df_train = df.drop(["participant", "category", "set"], axis=1)

# Split into features (X) and target (y) drop the column we want to predict
X = df_train.drop("label", axis=1)
y = df_train["label"]

# train-test split and esnure you stratify during data selections

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
    
)

# check the stratify in action and visualize the train_test_split ratios based on target column
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------


# we isolate the features that are derived from the different transformations applied to the original dataframe
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp" in f]
frequency_features = [f for f in df_train.columns if ("_freq_" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

# How many features are there in total per faetures type
print("Basic features: ", len(basic_features))

print("Square features: ", len(square_features))    

print("PCA features: ", len(pca_features))

print("Time features: ", len(time_features))

print("Frequency features: ", len(frequency_features))

print("Cluster features: ", len(cluster_features))


# Stacking features into different combinations of feature sets
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
 
# test the performance accuracy of the the features

learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features=max_features, X_train=X_train, y_train=y_train)

# Visualize selected features diminishing returns on accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores, marker="o", color="dodgerblue")

plt.xlabel("Number of features")    

plt.ylabel("Accuracy")

plt.title("Forward Selection")  

plt.xticks(np.arange(1, max_features + 1, 1))

plt.show()

# hardcode below for reference for 10 **selected_features** as a result of 
# the FeedForward Selection algorithm
print(selected_features)

selected_features = ['acc_y_temp_mean_ws_5',
 'duration',
 'acc_x_temp_mean_ws_5_freq_0.0_Hz_ws_14',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_y_temp_std_ws_5',
 'acc_x_temp_mean_ws_5_freq_2.5_Hz_ws_14',
 'gyr_y_freq_weighted',
 'acc_x_freq_0.357_Hz_ws_14',
 'acc_z_freq_2.5_Hz_ws_14',
 'acc_y']


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

# feedforward selected features added to the list of engineered features *selected_features*

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features    
]

feature_names = [   # this will be used to ease the plotting of features sets
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features"         # got as feature names from running the feedforward feature selection learner
]

iterations = 1 # number of iterations for grid search 

score_df = pd.DataFrame()

# --------------------------------
# GridSearch Code for Training 5 different classifiers/models

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])



# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df

score_df.sort_values(by="accuracy", ascending=False)

# Plot top see how each of the models and feature sets are performing on the test set
plt.figure(figsize=(10,10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()


# --------------------------------------------------------------
# Select best model 'RF' and evaluate results based on the best feature-set "feature_set_4"
# --------------------------------------------------------------

print("Select best model and evaluate results based on the *best feature* \n selected using the FeedFord select algo")
#Random Forest Model has the accuracy and pass best Feature set
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
    ) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
    )

# accuracy
accuracy = accuracy_score(y_test, class_test_y)
print(f"Printing Accuracy Score: {accuracy}")
type(class_test_prob_y)
class_test_prob_y.head()

# Generate a confusion matrix: To see which labels is our best model getting right vs wrong
    # pick the labels 
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Borrow code for cm labelling
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()    


# --------------------------------------------------------------
# Select train and test data based on participant exclusion to create unseen validation set
# --------------------------------------------------------------
    
# Load the dataset that exludes one participant data from the training set and
# test set will contain only that data of participant excluded. To test how we can generalise on unseen data of a gym app user.    
    # select one participant for testing and the rest for training
    
participant_df = df.drop(["set", "category"], axis=1)     #

# Train the model on a dataset that excludes one participant from the training dataset
# Participant A dropped out of dataset
X_train = participant_df[participant_df["participant"] != "A"].drop("label",axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

        ### Creating Unseen Validation Set  #######
# X-test dataset will be composed of participant A only data
X_test = participant_df[participant_df["participant"]=="A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"]== "A"]["label"]

# Visualize the new validation set created
print("Visualize the new validation set created by\n elimination of participant A from the training set to become the unseen validation set")

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="darkblue", label="Test")
plt.legend()
plt.show()



#Drop participant column from resulting dataset

X_train = X_train.drop("participant", axis=1)
X_test = X_test.drop("participant", axis=1)

print("Running RF model on properly curated Validation dataset")
#Random Forest Model used to see how well the model generalises on unseen data
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
    ) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
    )

# accuracy
accuracy = accuracy_score(y_test, class_test_y)
print(f"Accuracy score: {accuracy}")
type(class_test_prob_y)
class_test_prob_y.head()

# Generate a confusion matrix: To see which labels is our best model getting right vs wrong
    # pick the labels 
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Borrow code for cm labelling
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()    



# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

#Random Forest Model has the accuracy and pass best Feature set
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
    ) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
    )

# accuracy
accuracy = accuracy_score(y_test, class_test_y)
print(f"Accuracy score for Random Forest: {accuracy}")

# Generate a confusion matrix: To see which labels is our best model getting right vs wrong
    # pick the labels 
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Borrow code for cm labelling
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()    

# --------------------------------------------------------------
# Try a a more model with the selected features
# --------------------------------------------------------------
# rf is simpler model but we use now Nueral Network model with gridsearch turned off
    #based on the selected features -10 columns


#Neural Network Model has the accuracy and pass best Feature set
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
    ) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
    )

# accuracy
accuracy = accuracy_score(y_test, class_test_y)
print(f"Accuracy score for Neural Net with GridSearch off: {accuracy}")

# Generate a confusion matrix: To see which labels is our best model getting right vs wrong
    # pick the labels 
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Borrow code for cm labelling
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 