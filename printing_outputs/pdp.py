import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import LogNorm
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def x_formatter(x, pos):
    return f'{x / 1000:.1f}'  # Converts MPa to GPa

# Load the pretrained model
model = keras.models.load_model(
   '../output/lin/10_deg/Test_10_all_data/2layers_10dense_199999epochs_sgd_optimizer_silu_activation/best_model.h5')
#model = keras.models.load_model(
 #   '../output/log/10_deg/Test_10_all_data/2layers_20dense_199999epochs_rmsprop_optimizer_relu_activation/best_model.h5')
output_file_path = (f'../output/figures/pdp_plot_')

# Load the data
file_path = '../output/lin/10_deg/Test_10_all_data/2layers_10dense_199999epochs_sgd_optimizer_silu_activation/asphalt_10_deg_predicted.xlsx'
#file_path = '../output/log/10_deg/Test_10_all_data/2layers_20dense_199999epochs_rmsprop_optimizer_relu_activation/asphalt_10_deg_predicted.xlsx'
data = pd.read_excel(file_path)

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Extract 'Initial Stiffness' and 'Binder Content' columns
stiff = data['Initial stiffness (Mpa)'].values
bc = data['Binder content (%)'].values

# Define actual optimization
lin = 'lin'
log = 'log'
actual_opt = lin

# Load the MinMaxScaler that was fit during training
scaler = joblib.load('../new_scaler_10_deg.save')

# Define the ranges for 'Binder content' and 'Initial Stiffness'
binder_content_range = np.linspace(3.9, 5.1, 100)
stiff_range = np.linspace(6598, 17574, 100)

# Preset fixed strain, penetration, and air voids values
fixed_strain = 150
fixed_penetration = 60
fixed_av = 5.3   #5.3  # Fixed value for air voids

# Define PMB and SIL combinations for two plots
pmb_sil_combinations = [(1, 0), (0, 1)]

# Initialize global min and max values
global_min = np.inf
global_max = -np.inf

# Create a list to store reshaped predictions
temp_reshaped_predictions = []

# Create a 1x2 subplot structure
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.65, 7 * 0.65))

# Flatten axes array for easy iteration
axs = axs.flatten()

# Define levels for contour plots

levels_1 = np.logspace(np.log10(10000), np.log10(2000000), num=20)      # pro lin
levels_2 = np.logspace(np.log10(10000), np.log10(2000000), num=20)      # pro lin
#levels_1 = np.logspace(np.log10(100000), np.log10(2000000), num=20)     # pro log
#levels_2 = np.logspace(np.log10(100000), np.log10(2000000), num=20)     # pro log

# Initialize a list to store DataFrames for concatenation
df_list = []

# Loop through each PMB and SIL combination and corresponding axis
for i, (pmb_value, sil_value) in enumerate(pmb_sil_combinations):
    # Generate a meshgrid for 'Binder content' and 'Initial Stiffness'
    binder_content, initial_stiffness = np.meshgrid(binder_content_range, stiff_range)

    # Flatten the meshgrid matrices
    binder_content_flat = binder_content.flatten()
    initial_stiffness_flat = initial_stiffness.flatten()

    # Create repeated arrays for fixed parameters
    strain_fixed_array = np.full(binder_content_flat.shape, fixed_strain)
    penetration_fixed_array = np.full(binder_content_flat.shape, fixed_penetration)
    pmb_fixed_array = np.full(binder_content_flat.shape, pmb_value)
    sil_fixed_array = np.full(binder_content_flat.shape, sil_value)
    av_fixed_array = np.full(binder_content_flat.shape, fixed_av)

    # Stack the input parameters together for scaling
    model_inputs = np.stack([binder_content_flat, strain_fixed_array, av_fixed_array, penetration_fixed_array,
                              pmb_fixed_array, sil_fixed_array, initial_stiffness_flat], axis=1)

    # Scale the inputs
    model_inputs_scaled = scaler.transform(model_inputs)

    # Predict the 'Number of cycles' using the model
    predicted_cycles_scaled = model.predict(model_inputs_scaled).flatten()

    # Apply a lower limit to the predictions
    predicted_cycles_scaled = np.clip(predicted_cycles_scaled, 10000, 10000000)

    # Reshape the predictions to match the meshgrid shape
    predicted_cycles_reshaped = predicted_cycles_scaled.reshape(binder_content.shape)

    # Apply logarithmic transformation for better visualization
    predicted_cycles_reshaped_log = np.log(predicted_cycles_reshaped + 1)

    # Back-transform the predictions to original scale
    predicted_cycles_back_transformed = np.exp(predicted_cycles_reshaped_log) - 1

    # Store the reshaped predictions
    temp_reshaped_predictions.append(predicted_cycles_reshaped_log)

    # Update global min and max values
    global_min = min(global_min, predicted_cycles_back_transformed.min())
    global_max = max(global_max, predicted_cycles_back_transformed.max())

    # Add predictions to DataFrame list
    df_list.append(pd.DataFrame({
        'Binder Content (%)': binder_content_flat,
        'Initial Stiffness (MPa)': initial_stiffness_flat,
        'Predicted Cycles': predicted_cycles_scaled
    }))

    # Choose levels based on subplot index
    levels = levels_1 if i == 0 else levels_2

    # Plotting
    ax = axs[i]
    contourf = ax.contourf(initial_stiffness, binder_content, predicted_cycles_back_transformed,
                           levels=levels, cmap='jet', norm=LogNorm(vmin=levels.min(), vmax=levels.max()))

    contour = ax.contour(initial_stiffness, binder_content, predicted_cycles_back_transformed,
                         levels=levels, colors='k')
    ax.scatter(stiff, bc, color='white', marker='v', s=35, edgecolors='black', label='Data Points')

    # Custom Titles for Each Subplot
    if pmb_value == 1 and sil_value == 0:
        title = 'Binder type = PMB'
    elif pmb_value == 0 and sil_value == 1:
        title = 'Binder type = NB'
    else:
        title = rf'$\mathrm{{PMB}} = {pmb_value}, \mathrm{{SIL}} = {sil_value}$'

    # Titles and labels
    times_10 = r'\times 10^{-6}$'
    nf = r'$N_{\mathrm{f}}$'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'Initial stiffness (GPa)')
    ax.set_ylabel(r'Binder content (\%)')

    # Apply the formatter to x-axis to display in GPa
    ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

    # Set integer ticks for x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add a common colorbar for all subplots
plt.subplots_adjust(top=0.8, right=0.8)
cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.04])
#fig.colorbar(contourf, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.05, label=r'$N_{\mathrm{f}}$')

cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.05, label=r'$N_{\mathrm{f}}$')

# Set specific ticks on the colorbar
cbar.set_ticks([1e4, 5e4, 1e5, 5e5, 1e6, 2e6])  #pro lin
#cbar.set_ticks([1e5, 5e5, 1e6, 2e6])   # pro log

# Optionally, set custom tick labels if needed
cbar.set_ticklabels(['$1 \\times 10^4$', '$5 \\times 10^4$', '$1 \\times 10^5$', '$5 \\times 10^5$', '$1 \\times 10^6$', '$2 \\times 10^6$'])   #pro lin
#cbar.set_ticklabels(['$1 \\times 10^5$', '$5 \\times 10^5$', '$1 \\times 10^6$', '$2 \\times 10^6$'])   # pro log

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(output_file_path + actual_opt + '.pdf', format='pdf')
plt.show()

# After the loop, concatenate all DataFrames in the list
predicted_df = pd.concat(df_list, ignore_index=True)

# Display the predicted values in a table
print(predicted_df)

# Save the DataFrame to an Excel file
predicted_df.to_excel('../output/figures/predicted_values.xlsx', index=False)
