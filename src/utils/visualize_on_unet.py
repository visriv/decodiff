import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.spectrum import get_spatial_spectrum

def visualize_interm_embeds(interm_features_dict, save_dir, config):
    """
    Visualize intermediate outputs from Unet1 and Unet2 in a U-shape structure with symmetry.
    
    :param interm_features_dict: Dictionary containing intermediate outputs from Unet1 and Unet2.
           Each sub-dictionary has layer names as keys and tensors as values.
    """
    unet_keys = list(interm_features_dict.keys())
    num_unets = len(unet_keys)  # Number of Unets (2 in your case)

    # Initialize lists to categorize layers
    down_layers, mid_layers, up_layers = [], [], []

    # Categorize layers based on key names
    layer_names = list(interm_features_dict[unet_keys[0]].keys())  # Assuming both UNet1 and Unet2 have the same structure
    for layer_name in layer_names:
        if 'down' in layer_name.lower():
            down_layers.append(layer_name)
        elif 'up' in layer_name.lower():
            up_layers.append(layer_name)
        elif 'mid' in layer_name.lower():
            mid_layers.append(layer_name)

    num_down_layers = len(down_layers)
    num_mid_layers = len(mid_layers)
    num_up_layers = len(up_layers)

    # Calculate total number of columns needed:
    # Columns for downsampling, middle, and upsampling layers, plus separator columns
    num_columns = 1 + num_mid_layers + 1 + 1  # Extra column for separator
    # print('num_columns, num_mid_layers, num_unets:', num_columns, num_mid_layers, num_unets )
    fig, axes = plt.subplots(
        nrows=max(num_down_layers, num_up_layers, 1)+1,
        ncols=num_columns * num_unets + 1,
        figsize=(120, 60),  # Adjust figure size based on layout
    )
    fontsize = 16
    fig.suptitle(f"Intermediate Outputs for {', '.join(unet_keys)}", fontsize=18)

    def set_border(ax, layer_name):
        # print(layer_name)
        if layer_name.endswith('0'):
            color = 'yellow'
        elif layer_name.endswith('1'):
            color = 'green'
        elif layer_name.endswith('2'):
            color = 'cyan'
        elif layer_name.endswith('bridge'):
            color = 'purple'
        else:
            color = None

        if color:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)  # Set thick border






    im = None

    # Plotting for each UNet
    for u_idx, unet_key in enumerate(unet_keys):
        layer_dict = interm_features_dict[unet_key]

        # Plot Downsampling Layers in the first columns
        for i, layer_name in enumerate(down_layers):
            if layer_name in layer_dict:
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                ax = axes[i, u_idx * num_columns]  # Downsampling columns
                im = ax.imshow(feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                # ax.axis('off')
                set_border(ax, layer_name)  # Set border color


        # Plot Middle Layers in the next columns after down layers (center row)
        for i, layer_name in enumerate(mid_layers):
            if layer_name in layer_dict:
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                ax = axes[num_down_layers, u_idx * num_columns + i + 1]  # Middle layer columns
                im = ax.imshow(feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                # ax.axis('off')
                set_border(ax, layer_name)  # Set border color


        # Plot Upsampling Layers in the last columns, reverse order
        for i, layer_name in enumerate(reversed(up_layers)):
            if layer_name in layer_dict:
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                ax = axes[i, u_idx * num_columns + num_mid_layers + 1]  # Upsampling columns
                im = ax.imshow(feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                # ax.axis('off')
                set_border(ax, layer_name)  # Set border color


    # Add a common colorbar using the first imshow object (im)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('velocity component', fontsize=fontsize)
    
    # Add vertical lines between Unet1 and Unet2
    for ax in axes[:, num_columns - 1]:
        ax.axvline(x=ax.get_xlim()[1], color='black', linestyle='--', linewidth=2)  # Add vertical separator line

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the spacing for the suptitle
    fig.savefig(os.path.join(save_dir, 'interm_embeds_comparison.png'))
    plt.show()

# Example usage
# interm_features_dict = {
#     "Unet1": {
#         "down1": torch.randn(1, 3, 64, 64),
#         "down2": torch.randn(1, 3, 32, 32),
#         "mid": torch.randn(1, 3, 16, 16),
#         "up1": torch.randn(1, 3, 32, 32),
#         "up2": torch.randn(1, 3, 64, 64)
#     },
#     "Unet2": {
#         "down1": torch.randn(1, 3, 64, 64),
#         "down2": torch.randn(1, 3, 32, 32),
#         "mid": torch.randn(1, 3, 16, 16),
#         "up1": torch.randn(1, 3, 32, 32),
#         "up2": torch.randn(1, 3, 64, 64)
#     }
# }
# visualize_interm_embeds(interm_features_dict, save_dir='/path/to/save', config=None)




def visualize_spatial_spectra(interm_features_dict, save_dir, config):
    """
    Visualize sine waves in a U-shape structure with symmetry, keeping the same structure as intermediate feature visualization.
    
    :param interm_features_dict: Dictionary containing intermediate outputs from Unet1 and Unet2.
           Each sub-dictionary has layer names as keys, and we will plot sine waves instead of features.
    """
    unet_keys = list(interm_features_dict.keys())
    num_unets = len(unet_keys)  # Number of Unets (2 in your case)

    # Initialize lists to categorize layers
    down_layers, mid_layers, up_layers = [], [], []

    # Categorize layers based on key names
    layer_names = list(interm_features_dict[unet_keys[0]].keys())  # Assuming both UNet1 and Unet2 have the same structure
    for layer_name in layer_names:
        if 'down' in layer_name.lower():
            down_layers.append(layer_name)
        elif 'up' in layer_name.lower():
            up_layers.append(layer_name)
        elif 'mid' in layer_name.lower():
            mid_layers.append(layer_name)

    num_down_layers = len(down_layers)
    num_mid_layers = len(mid_layers)
    num_up_layers = len(up_layers)

    # Calculate total number of columns needed:
    num_columns = 1 + num_mid_layers + 1 + 1  # Extra column for separator
    print('num_columns, num_mid_layers, num_unets:', num_columns, num_mid_layers, num_unets)
    
    fig, axes = plt.subplots(
        nrows=max(num_down_layers, num_up_layers, 1)+1,
        ncols=num_columns * num_unets + 1,
        figsize=(120, 60),  # Adjust figure size based on layout
    )
    
    fontsize = 16
    fig.suptitle(f"Sine Wave Visualization for {', '.join(unet_keys)}", fontsize=18)

    def set_border(ax, layer_name):
        if layer_name.endswith('0'):
            color = 'yellow'
        elif layer_name.endswith('1'):
            color = 'green'
        elif layer_name.endswith('2'):
            color = 'cyan'
        elif layer_name.endswith('bridge'):
            color = 'purple'
        else:
            color = None

        if color:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)  # Set thick border

    # Function to generate a sine wave
    def generate_sine_wave(frequency, amplitude, num_points=1000):
        x = np.linspace(0, 2 * np.pi, num_points)
        y = amplitude * np.sin(frequency * x)
        return x, y

    # Plotting for each UNet
    for u_idx, unet_key in enumerate(unet_keys):
        layer_dict = interm_features_dict[unet_key]

        # Plot Downsampling Layers in the first columns
        for i, layer_name in enumerate(down_layers):
            if layer_name in layer_dict:
                ax = axes[i, u_idx * num_columns]  # Downsampling columns
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                energy_sorted, wavenumber_sorted = get_spatial_spectrum(feature_map)
                
                ax.plot(wavenumber_sorted, energy_sorted)
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1, top=500000) 
                set_border(ax, layer_name)  # Set border color

        # Plot Middle Layers in the next columns after down layers (center row)
        for i, layer_name in enumerate(mid_layers):
            if layer_name in layer_dict:
                ax = axes[num_down_layers, u_idx * num_columns + i + 1]  # Middle layer columns
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                energy_sorted, wavenumber_sorted = get_spatial_spectrum(feature_map)
                
                ax.plot(wavenumber_sorted, energy_sorted)
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1, top=500000) 
                set_border(ax, layer_name)  # Set border color

        # Plot Upsampling Layers in the last columns, reverse order
        for i, layer_name in enumerate(reversed(up_layers)):
            if layer_name in layer_dict:
                ax = axes[i, u_idx * num_columns + num_mid_layers + 1]  # Upsampling columns
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                energy_sorted, wavenumber_sorted = get_spatial_spectrum(feature_map)
                
                ax.plot(wavenumber_sorted, energy_sorted)
                ax.set_title(f'{unet_key} - {layer_name}', fontsize=fontsize)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1, top=500000) 
                set_border(ax, layer_name)  # Set border color

    # Add vertical lines between Unet1 and Unet2
    for ax in axes[:, num_columns - 1]:
        ax.axvline(x=ax.get_xlim()[1], color='black', linestyle='--', linewidth=2)  # Add vertical separator line

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the spacing for the suptitle
    fig.savefig(os.path.join(save_dir, 'unet_spectrum_comparison.png'))
    plt.show()

# Example usage
# interm_features_dict = {
#     "Unet1": {
#         "down1": None,
#         "down2": None,
#         "mid": None,
#         "up1": None,
#         "up2":
