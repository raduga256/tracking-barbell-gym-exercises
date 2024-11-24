import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import  KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(r"..\..\data\interim\02_outliers_removed_chauvenents.pkl")

predictor_columns = list(df.columns[:6])  # include all the numerical values cols in df

    #Changing and setting the figure/plotiing themes

mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100     # for the right resolution swhen exporting figures
mpl.rcParams["lines.linewidth"] = 2 

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

# using the pandas function for interpolate()
for col in predictor_columns:
    df[col] = df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"]==25] ["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()

    # check and calculate duration for one set
duration = df[df["set"]==25].index[-1] - df[df["set"]==25].index[0]
duration.seconds

    # creating duration for all sets in a col

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    
    duration = end - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
df[["duration"]].head(-5)  

    #Average duration of the sets
duration_df = df.groupby(["category"])["duration"].mean()    

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass  = LowPassFilter()

fs = 1000/200 # 200ms being our index interval
cutoff = 1      # (cutoff freq) more less the temperature/degree of the curve smoothening factor

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5) #
df_lowpass.info() # check new column name added

subset = df_lowpass[df_lowpass["set"] == 23]        #working with a subset of one set of exercises data file
print(subset["label"][0])

# Check for non-null values NaN before applying the LowPass filter
subset.isnull().sum()

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="row data")
    # Applying the filtering for smoothening
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=[0.5, 1.15], fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=[0.5, 1.15], fancybox=True, shadow=True)

    # apply lowfilter to all columns 
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    # overide the original df col with smoothened values
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    # delete created extra column now
    del df_lowpass[col + "_lowpass"]

df_lowpass.info() # check new column name added
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# make a copy of the dataset
df_pca = df_lowpass.copy()
pca_analysis = PrincipalComponentAnalysis()

#applying PCA transformation on all our predictor cols
pca_values = pca_analysis.determine_pc_explained_variance(df_pca, predictor_columns)

# Applying the elbow technique to identify the optimal number of cols that best capture the variance
pca_values

# visualize the explained variance ratio
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pca_values) + 1), pca_values, "bx-")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio vs. Principal Component")
plt.show()

# Applying the pca to our dataset with 3 principal components.
df_pca = pca_analysis.apply_pca(df_pca, predictor_columns, 3) 
df_pca.info()
# select one set of data for testing
subset = df_pca[df_pca["set"]==35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# This will be used to create new features of r -scalars for further analysis.
# --------------------------------------------------------------
#make a copy of the dataset
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

# create new columns for the sum of squares attributes

df_squared["acc_r"] = np.sqrt(acc_r) 
df_squared["gyr_r"] = np.sqrt(gyr_r) 

# visualize the sum of squares attributes for one set of exercises data file

subset = df_squared[df_squared["set"]==14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
df_temporal.info()

NumAbs = NumericalAbstraction()

# calculate the temporal abstraction for each column
# predictor columns must include the extra 2 columns for sum of squares values columns/features

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
predictor_columns

#window size, for ROLLING WINDOW for aggregation

window_size = int(1000/200) # 1000ms/200ms = 5 windows. 200ms was our index interval/step size

# loop through all the predictor columns and calculate the temporal abstraction

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "std") 
    #df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "max") 

df_temporal.info()
# To avoid having the new rolling cols features including data of teh different sets files for exercises,
# to apply correctly calculate the rolling features for each set file, we can use groupby() function on sets and 
# apply the rolling operation//function --->> Temporal Abstraction.abstract_numerical() function.

# make subsets of each of the sets and then compute the values for Temporal Abstraction.abstract_numerical() function.
df_temporal_list = []   # empty list to store/append the dataframes of each set file modified by Temporal Abstraction.abstract_numerical() function

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std") 

    # Creating a new modified dataframe :add the modified subset to the list
    df_temporal_list.append(subset)

# Concatenate all the modified subset dataframes to get the final dataset
df_temporal =  pd.concat(df_temporal_list)      # override the original df with the final dataset
df_temporal.info()  # we now have the final dataset with all the necessary features and less NaN values.

# Visualize: have at we have accomplished so far.
subset[["acc_y","acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()    #accelerometer y-axis over time 
subset[["gyr_y","gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features  - Advanced Decomposition of Time Series
# --------------------------------------------------------------
   
    # Aalways reset the index for each set file
df_freq = df_temporal.copy().reset_index()
df_freq.info()
df_freq.columns

FreqAbs = FourierTransformation()

# sampling rate - fs, in Hz

fs = int(1000/200) # 200ms being our index interval
ws = int(2800/200) # average length of a repeation - (2.8 seconds) divided by the sampling rate - fs

# applying the frequency transformation on only one of our predictor cols
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_x_temp_mean_ws_5"], ws, fs) # Try to decompose col "acc_y"
df_freq.info()  # check new column name added

# visualize the frequency attributes for one set of exercises data file as a result column ["acc_x_temp_mean_ws_5"] 
# decomposition

subset = df_freq[df_freq["set"]==15]
subset[["acc_x_temp_mean_ws_5"]].plot()  

# col ["acc_x_temp_mean_ws_5"] produces more features. select more cols from subset of features and plot results
subset.columns

subset[
        [
            
            "acc_x_temp_mean_ws_5_pse",
            "acc_x_temp_mean_ws_5_freq_0.714_Hz_ws_14",
            "acc_x_temp_mean_ws_5_freq_1.071_Hz_ws_14",
            "acc_x_temp_mean_ws_5_freq_2.5_Hz_ws_14"
        ]
    ].plot()

# Experiment with different frequency ranges to see how they affect the results.
subset[
        [
            "acc_x_temp_mean_ws_5_freq_weighted",
            "acc_x_temp_mean_ws_5_pse",
            "acc_x_temp_mean_ws_5_freq_0.714_Hz_ws_14",
            "acc_x_temp_mean_ws_5_freq_1.071_Hz_ws_14",
            "acc_x_temp_mean_ws_5_freq_2.5_Hz_ws_14"
        ]
    ].plot()

# Applying abstraction on all the predictor cols per each set file
# Then concatenate all the modified subset dataframes to get the final dataset

df_freq_list = []   # empty list to store/append the dataframes of each set file modified by FrequencyTransformation.abstract_frequency() function
df_freq.columns

for s in df_freq["set"].unique():
    print(f"Processing set {s} :: applying frequency transformation...")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

     

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)    # reset index back to original df with the final dataset by dropping the new index  
# Check for "epoch (ms)" set as the index and the dataset info

df_freq.info()
df_freq.columns
df_freq.index




# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()          # drop NaN values

# 50% overlap for each window because we want to preserve some temporal continuity in our data
df_freq = df_freq.iloc[::2] # select every second row to create overlapping windows

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"] # working with only accelerometer data for clustering
k_values = range(2, 10)
inertia = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)

# Plotting the inertia plot to find the optimal number of clusters
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertia, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Sum of squared distances')
plt.show()  

# Fitting the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=5, n_init=20, random_state=42) 
subset = df_cluster[cluster_columns]
cluster_labels = kmeans.fit_predict(subset)

# Adding the cluster labels to the original dataset

df_cluster['cluster'] = cluster_labels

# Plotting the clusters on a 3D scatter plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=c)
    
ax.set_xlabel('acc_x')

ax.set_ylabel('acc_y')

ax.set_zlabel('acc_z')

ax.legend()

plt.show()

# Subsetting by label and we visualize 3d scatter plot for each cluster
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for label in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == label]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=label)
    
ax.set_xlabel('acc_x')

ax.set_ylabel('acc_y')

ax.set_zlabel('acc_z')

ax.legend()

plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle(r'..\..\data\interim\03_data_features.pkl')



