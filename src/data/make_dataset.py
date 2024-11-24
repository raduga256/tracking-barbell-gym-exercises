import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read sample single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(r"..\..\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
    # use letter r"..\..\" for path escape sequence
    
single_file_gyr = pd.read_csv(r"..\..\data\raw\MetaMotion\MetaMotion\E-squat-heavy_MetaWear_2019-01-15T20.14.03.633_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
    # using glob package to read all files in a folder based on extensions

files = glob(r"..\..\data\raw\MetaMotion\MetaMotion\*csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
    #Extract portions of the filename string and add them as a column onto the data table
data_path = '..\\..\\data\\raw\\MetaMotion\\MetaMotion\\'
f = files[0]


    # split the filename at delimeter - to getsubcomponents. pick 1st subcomponent at index[0]
f.split("-")

    # We extracting 3 components off the filename 
paticipant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    # lets add extra columns to the original df
df = pd.read_csv(f)
df["paticipant"] = paticipant
df["label"] = label
df["category"] = category

df.head()   #df now has 3 extra columns



# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------


    # modify all files using 2 dfs and forloop
acc_df = pd.DataFrame()     # acc >> accelerometer
gyr_df = pd.DataFrame()     # gyr >> gyroscope
gyr_df.head(2)

    #counters for creating unique identifiers
acc_set = 1
gyr_set = 1

    # loop over all the files in the data folder

# Creating two giant dataframes for each file type with extra features/cols
# then we shall have to merge them later    
for f in files:
    
    # extract extract features into the dataframe
    paticipant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")  #Double right strip

    #print(category)
    # Read the all files into the dataframe
    df = pd.read_csv(f)
    
    # add the 3 extra columns of data
    df["paticipant"] = paticipant
    df["label"] = label
    df["category"] = category

    # filter and save/append the extra features into acc_df or gyr_df accordingly
    if "Accelerometer" in f:
        # create extra col to store meta data for each source data file/set into the df during the loop
        df["set"] = acc_set
        acc_set +=1
        acc_df = pd.concat([acc_df, df])
    
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set +=1
        gyr_df = pd.concat([gyr_df, df])





# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
    # convert the epoch(ms) a unix time  into pandas datetime format

acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms")
pd.to_datetime(df["time (01:00)"]).dt.month   # we can now apply datetime methods for extractions

    # set the index values to formmatted pd.datetime values for both our two dataframes
    # To have timestamps as index and afterwards delete original cols referencing time from dataframe
    
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms") 

# del for accelerometer dataframe
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

# del for gyroscope dataframe
del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]



# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
    # as a best practice to keep everything clean
   
files = glob(r"..\..\data\raw\MetaMotion\MetaMotion\*csv")
 
def read_data_from_files(files):
    
    acc_df = pd.DataFrame()     # acc >> accelerometer
    gyr_df = pd.DataFrame()     # gyr >> gyroscope

    #counters for creating unique identifiers
    acc_set = 1
    gyr_set = 1

    # loop over all the files in the data folder

    # Creating two giant dataframes for each file type with extra features/cols
    # then we shall have to merge them later    
    for f in files:
        
        # extract extract features into the dataframe
        paticipant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")  #Double right strip

        #print(category)
        # Read the all files into the dataframe
        df = pd.read_csv(f)
        
        # add the 3 extra columns of data
        df["paticipant"] = paticipant
        df["label"] = label
        df["category"] = category

        # filter and save/append the extra features into acc_df or gyr_df accordingly
        if "Accelerometer" in f:
            # create extra col to store meta data for each source data file/set into the df during the loop
            df["set"] = acc_set
            acc_set +=1
            acc_df = pd.concat([acc_df, df])
        
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set +=1
            gyr_df = pd.concat([gyr_df, df])
            
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms") 

    # del for accelerometer dataframe
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    # del for gyroscope dataframe
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

        # return the acc and gyr dataframes
    return acc_df, gyr_df

# Test code to define acc and gyro dataframes
acc_df, gyr_df = read_data_from_files(files)
acc_df.head(2)
gyr_df.head(2)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

    # we merge dataframes along the column axis so that 1 row will have both acc and gyr data
    # merge while avoiding duplicate columns. redundant data
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1) 

# for some rows a data is missing
# rename cols to easily identify acc and gyroscope data in the dataframe
data_merged.info()
data_merged.head(2)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",  
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
        # gyroscope records data at a higher frequency so we need fix off sets
        # specify aggregation methods e.g mean for pandas resample       
        
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

data_merged.sort_index()     # group datetime index for every 200ms as frequency
data_merged[:100].resample(rule="200ms").mean(numeric_only=True) # work on numerical cols

    #Resampling to Include Categorical columns data.
    # create a dictionary for the columns and pass aggregation method or instruction as dict value

data_merged.columns

sampling = {
    'acc_x':"mean", 
    'acc_y':"mean", 
    'acc_z':"mean", 
    'gyr_x':"mean", 
    'gyr_y':"mean", 
    'gyr_z':"mean", 
    'participant':"last",  #fill grouped by last value for cat data
    'label':"last", 
    'category':"last", 
    'set':"last"
}

# Select only 1000 rows to work with because original data is too big for our compute
data_merged[:1000].resample(rule="200ms").apply(sampling)

    # Don't run the above on entire dataframe ... many records will cause memory to fail
    # Split the data merged into dataframes per day frequency
# Create a list of dataframes for each day of the months
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
days[8][:300]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled.info() #fix set col datatupe to int
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

        # Export as pickle file and not as csv becuase we still going to use the data among different python files 
        # on the project.  pickle serialisation preserves data identify and struct. saves space
        # Export to folder data/interim
        
data_resampled.to_pickle(r"..\..\data\interim\01_data_processed.pkl")


