import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define activations and optimizers
activations = ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'SiLU', 'SiLU', 'SiLU', 'SiLU', 'GELU', 'GELU', 'GELU', 'GELU', 'Mish', 'Mish', 'Mish', 'Mish']
optimizers = ['Adam', 'AdamW', 'RMSprop', 'SGD', 'Adam', 'AdamW', 'RMSprop', 'SGD', 'Adam', 'AdamW', 'RMSprop', 'SGD', 'Adam', 'AdamW', 'RMSprop', 'SGD']

# R^2 scores for linear and logarithmic models
log_R2 = [0.158061371, 0.188584258, 0.279362957, 0.258801956, 0.114568871, 0.051513145, 0.164984759, 0.151912012,
          0.028733544, 0.022529898, 0.100670842, 0.118767482, 0.129339936, 0.058693534, 0.063773602, 0.08481047]
lin_R2 = [-0.043660753, -0.1260353, -0.063711342, -0.038891251, 0.208081312, 0.253841762, 0.292348811, 0.352456056,
          0.276977043, 0.289017919, 0.288835903, 0.289614117, 0.325912167, 0.293812, 0.306234772, 0.255088117]

# Colors for optimizers
colors = ['r', 'g', 'b', 'y']

# Unique activations for grouping
unique_activations = sorted(set(activations), key=activations.index)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 5 * 0.75))  # 1 row, 2 columns

bar_width = 0.25  # width of the bars
indices = np.arange(len(unique_activations))  # group positions

# Plotting bars for each optimizer
for i, optimizer in enumerate(sorted(set(optimizers), key=optimizers.index)):
    # Filter R2 scores by optimizer
    lin_r2_by_optimizer = [lin_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]
    log_r2_by_optimizer = [log_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]

    # Left subplot for linear RMSE criterion
    axs[0].bar(indices + i * bar_width - 0.5 * bar_width, lin_r2_by_optimizer, color=colors[i], width=bar_width, label=optimizer)

    # Right subplot for logarithmic RMSE criterion
    axs[1].bar(indices + i * bar_width - 0.5 * bar_width, log_r2_by_optimizer, color=colors[i], width=bar_width, label=optimizer)

# Adjusting the plots
for ax in axs:
    ax.set_ylabel(r'$\overline{R^2}$')
    ax.set_xticks(indices + bar_width / 2)  # Center the tick under the group of bars
    ax.set_xticklabels(unique_activations, ha="center")  # Set x-tick labels
    ax.axhline(0, color='black', linewidth=0.8)  # Add a line at y=0 for clarity

# Titles for subplots
axs[0].set_title(r'$L_{\mathrm{MSE}}(y, \hat{y})$')
axs[1].set_title(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')

# Common legend
plt.subplots_adjust(top=0.95)
handles, labels = axs[0].get_legend_handles_labels()  # Assuming handles and labels are the same for both subplots
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1))

# Adjust layout to fit the legend
fig.tight_layout(rect=[0, 0, 1, 0.90])

# Save the figure
output_file_path = '../output/figures/r2_score_different_activations_and_optimizers.pdf'
plt.savefig(output_file_path, format='pdf')
plt.show()