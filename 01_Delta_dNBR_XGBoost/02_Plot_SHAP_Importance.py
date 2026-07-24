import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'

    biome_dirs = r"Figure_results"
    biomes = [1, 2, 3, 4, 5, 12]
    features = ['AGB_diff', 'NDVI_diff', 'LFMC_diff', 'VPD_diff', 'Sc_diff', 'Rd_diff']

    feature_labels = {
        'AGB_diff': 'ΔAGB',
        'NDVI_diff': 'ΔNDVI',
        'LFMC_diff': 'ΔLFMC',
        'VPD_diff': 'ΔVPD',
        'Sc_diff': 'ΔSc',
        'Rd_diff': 'ΔRd'
    }

    biome_abbr = {
        1: "TrMFBF", 2: "TrDFBF", 3: "TrSCF", 4: "TeBMF",
        5: "TeCF", 6: "BoFT", 12: "MedFWS", 0: "All Biomes"
    }

    color1 = '#fcf1ef'
    color2 = color1

    def group_feature(feat_name):
        return feat_name in ['VPD_diff', 'LFMC_diff', 'NDVI_diff']

    bar_height = 0.6
    edge_color = '#2c3e50'
    value_offset_ratio = 0.025

    font_main_label = {'fontsize': 7, 'fontweight': 'normal'}
    font_tick_label = {'fontsize': 7, 'fontweight': 'normal'}
    font_bar_value = {'fontsize': 7, 'fontweight': 'bold'}
    font_biome_abbr = {'fontsize': 7, 'fontweight': 'bold', 'style': 'italic'}

    for biome in biomes:
        print(f"Processing Biome {biome}...")

        biome_dir = os.path.join(biome_dirs, f"Biome_{biome}")
        shap_path = os.path.join(biome_dir, "shap_values.csv")
        pred_path = os.path.join(biome_dir, "predictions.csv")

        if not os.path.exists(shap_path) or not os.path.exists(pred_path):
            print(f"  Warning: Missing data for Biome {biome}, skipping.")
            continue

        shap_df = pd.read_csv(shap_path)
        pred_df = pd.read_csv(pred_path)

        if 'orig_index' in shap_df.columns:
            shap_df = shap_df.set_index('orig_index')
        if 'orig_index' in pred_df.columns:
            pred_df = pred_df.set_index('orig_index')
        if not shap_df.index.equals(pred_df.index):
            shap_df = shap_df.reset_index(drop=True)
            pred_df = pred_df.reset_index(drop=True)

        mean_abs_shap = shap_df[features].abs().mean().sort_values(ascending=True)
        sorted_feats = mean_abs_shap.index.tolist()
        values = mean_abs_shap.values

        bar_colors = [color1 if group_feature(f) else color2 for f in sorted_feats]

        fig, ax = plt.subplots(figsize=(1.2, 2.4))
        plt.subplots_adjust(left=0.36, right=0.82, top=0.98, bottom=0.15)

        y_pos = np.arange(len(sorted_feats))
        bars = ax.barh(y_pos, values, height=bar_height,
                       color=bar_colors, edgecolor=edge_color, linewidth=0.8, zorder=3)

        ax.barh(
            y_pos, values,
            height=bar_height,
            color='none',
            edgecolor=edge_color,
            hatch='///',
            linewidth=0.0,
            zorder=4
        )


        offset = value_offset_ratio * values.max()
        for i, val in enumerate(values):
            ax.text(val + offset, i, f"{val:.3f}",
                    va='center', ha='left', **font_bar_value)

        display_names = [feature_labels.get(f, f.replace('_', ' ')) for f in sorted_feats]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, **font_tick_label)

        ax.set_xlabel('Mean |SHAP value|', **font_main_label, labelpad=4)

        ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=0)

        ax.tick_params(axis='x', labelsize=7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)


        save_path = os.path.join(biome_dir, "mean_shap_bar.png")
        plt.savefig(save_path, dpi=500)
        plt.close(fig)

        print(f"✅ Saved Biome {biome}: {save_path}")


if __name__ == '__main__':
    main()
