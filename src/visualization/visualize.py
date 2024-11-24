import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(r"..\..\data\interim\01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
    # start for by filtering df to only plot each set of exercises

set_df = df[df["set"] == 1] 
set_df.info()
    # plot a single column
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True)) # Reveals number of samples 

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

    # create multiple subsets dataframes from all labels ... using unique()
for label in df["label"].unique():
    subset = df[df["label"]==label]  # loop over all the labels

    # Plot the subset ...create figure first
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
    
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
    # Tweaking the figures to get better idea of how patterns develope over time
    # changing/customize/set rcParams ... set a sytle

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100     # for the right resolution swhen exporting figures

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
        # creating subset dataframes using query() method
category_df = df.query("label == 'squat'")
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

        # Grouped plots
fig, ax = plt.subplots()        
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Compare participants    A vs B
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()        
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis    ... the x,y,z for both accelerometer and gyroscope data
# --------------------------------------------------------------

label = "squat"
participant = "A"

all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
all_axis_df.columns
#plot the filter dataframe with multiple axis
fig, ax = plt.subplots()        
all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
  
    
# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# Plots for data from the Accelerometer
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index()
                       )
        
        # Filter out plots without data. Incase a participant didn't perform a certain
        # type of exercise
        if len(all_axis_df) > 0:
        
            #plot the filter dataframe with multiple axis
            fig, ax = plt.subplots()        
            all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
        

# Plots for data from the Gyroscope
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index()
                       )
        
        # Filter out plots without data. Incase a participant didn't perform a certain
        # type of exercise
        if len(all_axis_df) > 0:
        
            #plot the filter dataframe with multiple axis
            fig, ax = plt.subplots()        
            all_axis_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"

combined_plot_df = (df.query(f"label == '{label}'")
               .query(f"participant == '{participant}'")
               .reset_index(drop=True)
) 

#plot the accelerometer and Gyroscope data on the same figure
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))        
combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

# Style legends individually on each axis
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# Plots for data from the Accelerometer
for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
                       .query(f"participant == '{participant}'")
                       .reset_index()
                       )
        
        # Filter out plots without data. Incase a participant didn't perform a certain
        # type of exercise
        if len(combined_plot_df) > 0:

            #plot the accelerometer and Gyroscope data on the same figure
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))        
            combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
            combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

            # Style legends individually on each axis
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

            ax[1].set_xlabel("samples")

            # Export the figures
            plt.savefig(f"..\\..\\reports\\figures\{label.title()}({participant}).png")
            plt.show()








