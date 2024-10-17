import matplotlib.pyplot as plt
import numpy as np
import os




def visualize_interm_embeds(interm_features_dict,
                            save_dir,
                            config
                            ):
    """
    Visualize intermediate outputs from Unet1 and Unet2.
    
    :param interm_features_dict: Dictionary containing intermediate outputs from Unet1 and Unet2.
           Each sub-dictionary has layer names as keys and tensors as values.
    """
    # Iterate through Unet1 and Unet2
    for unet_key, layer_dict in interm_features_dict.items():
        print(f"Visualizing outputs for: {unet_key}")
        
        # Create a figure for the UNet
        num_layers = len(layer_dict)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))
        fig.suptitle(f"Intermediate Outputs for {unet_key}", fontsize=18)
        
        # Iterate through layers and their outputs
        for idx, (layer_name, output) in enumerate(layer_dict.items()):
            output = output[0]  # Pick the first item from the batch (if batch size > 1)
            feature_map = output.detach().cpu().numpy()
            
            num_channels = feature_map.shape[0]
            ax = axes[idx] if num_layers > 1 else axes  # In case there's only 1 layer
            
            # If it's a multi-channel feature map, visualize the first channel (for simplicity)
            ax.imshow(feature_map[0], cmap='RdBu_r')  # You can adjust to visualize multiple channels if needed
            ax.set_title(f'{unet_key} - {layer_name}', fontsize=12)
            ax.axis('off')

        # Adjust layout and display
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the spacing for the suptitle
        fig.savefig(os.path.join(save_dir, 'interm_embeds.png'))
        plt.show()


# Example usage
# Let's assume you have a dictionary structured as:
# interm_features_dict = {
#     "Unet1": {
#         "layer1": tensor1, 
#         "layer2": tensor2
#     },
#     "Unet2": {
#         "layer1": tensor3, 
#         "layer2": tensor4
#     }
# }
# The function will plot all these layers as subplots.




import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_interm_embeds1(interm_features_dict, save_dir, config):
    """
    Visualize intermediate outputs from Unet1 and Unet2 in a U-shape structure.
    
    :param interm_features_dict: Dictionary containing intermediate outputs from Unet1 and Unet2.
           Each sub-dictionary has layer names as keys and tensors as values.
    """
    unet_keys = list(interm_features_dict.keys())
    num_unets = len(unet_keys)  # Number of Unets (2 in your case)

    # Initialize lists to categorize layers
    down_layers, mid_layers, up_layers = [], [], []

    # Categorize layers based on key names
    layer_names = list(interm_features_dict[unet_keys[0]].keys())  # Assuming both UNet1 and UNet2 have the same structure
    for layer_name in layer_names:
        if 'down' in layer_name.lower():
            down_layers.append(layer_name)
        elif 'up' in layer_name.lower():
            up_layers.append(layer_name)
        elif 'mid' in layer_name.lower():
            mid_layers.append(layer_name)

    # Create a large grid for subplots (rows for downsampling/up/mid, columns for Unet1 and Unet2)
    num_down_layers = len(down_layers)
    num_mid_layers = len(mid_layers)
    num_up_layers = len(up_layers)
    
    fig, axes = plt.subplots(
        nrows=max(num_down_layers, num_mid_layers, num_up_layers),
        ncols=num_unets * 3,  # 3 columns: down, mid, up for both Unets
        figsize=(15, (num_down_layers + num_mid_layers + num_up_layers) * 2),
        gridspec_kw={'width_ratios': [1, 1, 1] * num_unets}
    )
    
    fig.suptitle(f"Intermediate Outputs for {', '.join(unet_keys)}", fontsize=18)

    # Plotting for each UNet
    for u_idx, unet_key in enumerate(unet_keys):
        layer_dict = interm_features_dict[unet_key]

        # Plot Downsampling Layers
        for i, layer_name in enumerate(down_layers):
            if layer_name in layer_dict:
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                ax = axes[i, u_idx * 3]  # Downsampling column
                ax.imshow(feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name} (Down)', fontsize=12)
                ax.axis('off')

        # Plot Middle Layers
        for i, layer_name in enumerate(mid_layers):
            if layer_name in layer_dict:
                mid_output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                mid_feature_map = mid_output[0]  # Visualize the first channel
                ax = axes[i, u_idx * 3 + 1]  # Middle column
                ax.imshow(mid_feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name} (Mid)', fontsize=12)
                ax.axis('off')

        # Plot Upsampling Layers
        for i, layer_name in enumerate(up_layers):
            if layer_name in layer_dict:
                output = layer_dict[layer_name][0].detach().cpu().numpy()  # Select the first batch element
                feature_map = output[0]  # Visualize the first channel
                ax = axes[i, u_idx * 3 + 2]  # Upsampling column
                ax.imshow(feature_map, cmap='RdBu_r')
                ax.set_title(f'{unet_key} - {layer_name} (Up)', fontsize=12)
                ax.axis('off')

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
