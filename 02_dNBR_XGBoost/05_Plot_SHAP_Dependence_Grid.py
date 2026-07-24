import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from pygam import LinearGAM, s


def main() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'

                                 
          
                                 
    biome_dirs = r"Figure_results"
    biomes = [1, 2, 3, 4, 5, 6, 12]
    features = ['AGB', 'NDVI', 'LFMC', 'VPD', 'Sc', 'Rd']

    feature_labels = {
        'AGB': 'AGB (Mg ha⁻¹)',                  
        'Sc': 'Sc',                          
        'Rd': 'Rd (m km⁻²)',                             
        'VPD': 'VPD (kPa)',                    
        'LFMC': 'LFMC (%)',
        'NDVI': 'NDVI'

    }

          
    p_a_color = "#43b0f1"
    non_p_a_color ="#ffca3a"

    smooth_color = '#d9042b'

                                 
         
                                 
    for biome in biomes:
        print(f"Processing Biome {biome}...")

        biome_dir = os.path.join(biome_dirs, f"Biome_{biome}")
        groups = {
            'PAs': os.path.join(biome_dir, "PAs"),
            'nonPAs': os.path.join(biome_dir, "nonPAs")
        }

        data_dict = {}

        for group_name, folder in groups.items():
            shap_path = os.path.join(folder, "shap_values.csv")
            pred_path = os.path.join(folder, "predictions.csv")

            if not os.path.exists(shap_path) or not os.path.exists(pred_path):
                print(f"  Warning: Missing data for {group_name} in Biome {biome}, skipping.")
                continue

            shap_df = pd.read_csv(shap_path)
            pred_df = pd.read_csv(pred_path)

                             
            if 'orig_index' not in shap_df.columns:
                shap_df['orig_index'] = shap_df.index
            if 'orig_index' not in pred_df.columns:
                pred_df['orig_index'] = pred_df.index

            shap_df = shap_df.set_index('orig_index')
            pred_df = pred_df.set_index('orig_index')


            data_dict[group_name] = (pred_df, shap_df)

                                     
                          
                                     
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(7.1, 4.2))
                
        plt.subplots_adjust(left=0.07, right=0.93, top=0.985, bottom=0.075, hspace=0.35, wspace=0.075)


        subplot_labels = list(string.ascii_lowercase)
        label_idx = 2

        for i in range(3):             
            for j in range(2):              
                feat_idx = i * 2 + j
                if feat_idx >= len(features):
                    continue

                feat = features[feat_idx]
                feat_label = feature_labels[feat]

                                       
                                   
                                       
                ax_left = axes[i, j * 2]

                if 'PAs' in data_dict:
                    pred_df, shap_df = data_dict['PAs']
                    x_vals = pred_df[feat].values
                    y_vals = shap_df[feat].values

                    ax_left.scatter(
                        x_vals, y_vals,
                        s=25, alpha=0.6,
                        edgecolors='k', linewidths=0.4,
                        zorder=2,
                        color=p_a_color
                    )

                                                      
                    x_min = np.min(x_vals)
                    x_max = np.max(x_vals)
                    x_range = x_max - x_min

                    lower_bound = x_min + 0.025 * x_range
                    upper_bound = x_max - 0.025 * x_range
                    mask = (x_vals >= lower_bound) & (x_vals <= upper_bound)

                    x_pa_f = x_vals[mask]
                    y_pa_f = y_vals[mask]

                            
                    if len(x_pa_f) > 10:
                        gam_pa = LinearGAM(s(0, n_splines=10, lam=2)).fit(
                            x_pa_f.reshape(-1, 1),
                            y_pa_f
                        )

                        x_smooth_pa = np.linspace(x_pa_f.min(), x_pa_f.max(), 200)
                        y_smooth_pa = gam_pa.predict(x_smooth_pa)

                        ax_left.plot(
                            x_smooth_pa, y_smooth_pa,
                            color=smooth_color,
                            linewidth=1.75,
                            zorder=3
                        )

                    ax_left.text(
                        0.97, 0.95, 'PAs',
                        transform=ax_left.transAxes,
                        fontsize=8, fontweight='bold',
                        va='top', ha='right',
                        color='black'
                    )
                else:
                    ax_left.axis('off')
                    continue

                                       
                                      
                                       
                ax_right = axes[i, j * 2 + 1]

                if 'nonPAs' in data_dict:
                    pred_df, shap_df = data_dict['nonPAs']
                    x_vals = pred_df[feat].values
                    y_vals = shap_df[feat].values

                    ax_right.scatter(
                        x_vals, y_vals,
                        s=25, alpha=0.6,
                        edgecolors='k', linewidths=0.4,
                        zorder=2,
                        color=non_p_a_color
                    )

                                                         
                    x_min = np.min(x_vals)
                    x_max = np.max(x_vals)
                    x_range = x_max - x_min

                    lower_bound = x_min + 0.025 * x_range
                    upper_bound = x_max - 0.025 * x_range
                    mask = (x_vals >= lower_bound) & (x_vals <= upper_bound)

                    x_non_f = x_vals[mask]
                    y_non_f = y_vals[mask]

                              
                    if len(x_non_f) > 10:
                        gam_non = LinearGAM(s(0, n_splines=10, lam=2)).fit(
                            x_non_f.reshape(-1, 1),
                            y_non_f
                        )

                        x_smooth_non = np.linspace(x_non_f.min(), x_non_f.max(), 200)
                        y_smooth_non = gam_non.predict(x_smooth_non)

                        ax_right.plot(
                            x_smooth_non, y_smooth_non,
                            color=smooth_color,
                            linewidth=1.75,
                            zorder=3
                        )

                    ax_right.text(
                        0.97, 0.95, 'Non-PAs',
                        transform=ax_right.transAxes,
                        fontsize=8, fontweight='bold',
                        va='top', ha='right',
                        color='black'
                    )
                else:
                    ax_right.axis('off')

                                       
                                              
                                       
                if 'PAs' in data_dict and 'nonPAs' in data_dict:
                    y_min = min(
                        data_dict['PAs'][1][feat].min(),
                        data_dict['nonPAs'][1][feat].min()
                    )
                    y_max = max(
                        data_dict['PAs'][1][feat].max(),
                        data_dict['nonPAs'][1][feat].max()
                    )

                    y_range = y_max - y_min
                    y_pad = 0.25 * y_range                           

                    for ax in [ax_left, ax_right]:
                        ax.set_ylim(y_min - y_pad, y_max + y_pad)
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                                       
                                       
                                       
                for ax, col in zip([ax_left, ax_right], [j * 2, j * 2 + 1]):

                    if col == 0:                  
                        ax.yaxis.set_ticks_position('left')
                        ax.yaxis.set_label_position('left')
                        ax.set_ylabel("SHAP value", fontsize=9)
                        ax.tick_params(
                            axis='y',
                            labelsize=8,           
                            length=2.8,         
                            width=1,         
                            pad=1.5             
                        )
                        ax.yaxis.set_label_coords(-0.25, 0.5)        

                    elif col == 1:                     
                        ax.yaxis.set_ticks_position('left')
                        ax.set_yticklabels([])
                        ax.set_ylabel("")

                    elif col == 2:                     
                        ax.yaxis.set_ticks_position('right')
                        ax.set_yticklabels([])
                        ax.set_ylabel("")

                    elif col == 3:                  
                        ax.yaxis.set_ticks_position('right')
                        ax.yaxis.set_label_position('right')
                        ax.set_ylabel("SHAP value", fontsize=9)
                        ax.tick_params(
                            axis='y',
                            labelsize=8,           
                            length=2.8,         
                            width=1,         
                            pad=1.5             
                        )
                        ax.yaxis.set_label_coords(1.25, 0.5)        

                    ax.set_xlabel(feat_label, fontsize=9, labelpad=0.5)
                    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

                                           
                                           
                                           
                    x_vals_all = ax.collections[0].get_offsets()[:, 0]
                    x_min, x_max = np.min(x_vals_all), np.max(x_vals_all)

                    x_range = x_max - x_min
                    x_pad = 0.05 * x_range                           

                    ax.set_xlim(x_min - x_pad, x_max + x_pad)

                                     
                    if x_range > 5:             
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                    else:            
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    ax.tick_params(
                        axis='x',
                        labelsize=8,           
                        length=2.8,         
                        width=1,         
                        pad=1.5             
                    )

                                       
                                     
                                       
                               
                                                                  
                                                  
                                                     
                                             
                   
                 
                                
                                                                      
                                                   
                                                     
                                             
                   

                label_idx += 2

                                     
              
                                     
        save_path = os.path.join(
            biome_dir,
            "dependence_features_PAs_vs_nonPAs_3x4_labeled.png"
        )

                           
        plt.savefig(save_path, dpi=1200)
        plt.close(fig)

        print(f"✅ Saved Biome {biome}: {save_path}")


if __name__ == '__main__':
    main()
