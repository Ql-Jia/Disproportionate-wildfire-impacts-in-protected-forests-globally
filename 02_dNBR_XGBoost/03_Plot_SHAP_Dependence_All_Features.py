import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import string
from pygam import LinearGAM, s


def main() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'

                                 
    biome_dirs = r"Figure_results"
    biomes = [1, 2, 3, 4, 5, 6, 12]
                          
    features = ['AGB', 'Sc', 'Rd', 'VPD', 'LFMC', 'NDVI']

    feature_labels = {
        'AGB': 'AGB (Mg ha⁻¹)',                  
        'Sc': 'Sc',                          
        'Rd': 'Rd (m km⁻²)',                             
        'VPD': 'VPD (kPa)',                    
        'LFMC': 'LFMC (%)',
        'NDVI': 'NDVI'

    }

    p_a_color = '#D55E00'       
    non_p_a_color = '#0072B2'
               
    frac_pa = 0.4
    frac_nonpa = 0.5
    line_width = 2
             
    tick_pad = 1.4                   
    tick_width = 0.9             
    tick_length = 2.4
    y_label_offset = -0.32                    


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

        if len(data_dict) < 2:
            print(f"  Skipping Biome {biome}: both PAs and nonPAs data required.")
            continue

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(2.8, 4))             
        plt.subplots_adjust(left=0.15, right=0.985, top=0.99, bottom=0.07,
                            hspace=0.365, wspace=0.35)

        subplot_labels = list(string.ascii_lowercase)

        for idx, feat in enumerate(features):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            pred_pa, shap_pa = data_dict['PAs']
            pred_non, shap_non = data_dict['nonPAs']

            x_pa = pred_pa[feat].values
            y_pa = shap_pa[feat].values
            x_non = pred_non[feat].values
            y_non = shap_non[feat].values


            x_min_pa, x_max_pa = np.min(x_pa), np.max(x_pa)
            x_range_pa = x_max_pa - x_min_pa
            lower_pa = x_min_pa + 0.025 * x_range_pa
            upper_pa = x_max_pa - 0.025 * x_range_pa
            mask_pa = (x_pa >= lower_pa) & (x_pa <= upper_pa)
            x_pa_f = x_pa[mask_pa]
            y_pa_f = y_pa[mask_pa]

                                                      
            gam_pa = LinearGAM(s(0, n_splines=10, lam=2)).fit(x_pa_f, y_pa_f)
                             
            x_smooth_pa = np.linspace(x_pa_f.min(), x_pa_f.max(), 200)
            y_smooth_pa = gam_pa.predict(x_smooth_pa)

                           
            x_min_non, x_max_non = np.min(x_non), np.max(x_non)
            x_range_non = x_max_non - x_min_non
            lower_non = x_min_non + 0.025 * x_range_non
            upper_non = x_max_non - 0.025 * x_range_non
            mask_non = (x_non >= lower_non) & (x_non <= upper_non)
            x_non_f = x_non[mask_non]
            y_non_f = y_non[mask_non]

            gam_non = LinearGAM(s(0, n_splines=10, lam=2)).fit(x_non_f, y_non_f)
            x_smooth_non = np.linspace(x_non_f.min(), x_non_f.max(), 200)
            y_smooth_non = gam_non.predict(x_smooth_non)

                         
            ax.plot(x_smooth_pa, y_smooth_pa,
                    color=p_a_color, linewidth=line_width, label='PAs', zorder=3)
            ax.plot(x_smooth_non, y_smooth_non,
                    color=non_p_a_color, linewidth=line_width, label='Non-PAs', zorder=3)


                    
            y_all = np.concatenate([y_pa, y_non])
            y_min, y_max = np.min(y_all), np.max(y_all)
            y_range = y_max - y_min
            y_pad = 0.2 * y_range
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

                  
            x_all = np.concatenate([x_pa, x_non])
            x_min, x_max = np.min(x_all), np.max(x_all)
            x_range = x_max - x_min
            x_pad = 0.05 * x_range
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

                  
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if x_range < 5:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                   
            ax.set_xlabel(feature_labels.get(feat, feat), fontsize=7.5)
            ax.xaxis.set_label_coords(0.5, -0.17)
            if col == 0:
                ax.set_ylabel("SHAP value", fontsize=7.5)
                                
                ax.yaxis.set_label_coords(y_label_offset, 0.5)
            else:
                ax.set_ylabel("")

                
            ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

                        
            if idx == 0:
                ax.legend(
                    loc='center',                                                
                    bbox_to_anchor=(0.51, 0.885),
                    fontsize=7,        
                    frameon=False,                      
                    framealpha=0.8,              
                    edgecolor='gray',        
                    fancybox=False,          
                    handlelength=0.8,          
                    handletextpad=0.3,           
                    borderpad=0.4,         
                    labelspacing=0.1,              
                    columnspacing=0.9,
                    ncol=2                    
                )

                             
            ax.tick_params(axis='both', labelsize=7, pad=tick_pad, width=tick_width, length=tick_length)

            
        save_path = os.path.join(biome_dir, "trend_lines_PAs_vs_nonPAs_3x2.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)

        print(f"✅ Saved Biome {biome}: {save_path}")


if __name__ == '__main__':
    main()
