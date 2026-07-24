                       
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"

                                
        
                                
    biomes = [1, 2, 3, 4, 5, 6, 12]
    root_path = 'Figure_results'

    color_nonpa ="#D55E00"
    color_pa = "#2C6DA4"


                                
          
                                
    def calc_metrics(y_true, y_pred):
        r, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r, rmse


                                
         
                                
    for biome in biomes:
        print(f"Processing Biome {biome}...")

        pa_file = os.path.join(root_path, f"Biome_{biome}", "PAs", "predictions.csv")
        nonpa_file = os.path.join(root_path, f"Biome_{biome}", "nonPAs", "predictions.csv")

        if not os.path.exists(pa_file) or not os.path.exists(nonpa_file):
            print(f"  ⚠ Missing files for Biome {biome}, skip.")
            continue

                                    
              
                                    
        df_pa = pd.read_csv(pa_file)
        df_nonpa = pd.read_csv(nonpa_file)

        y_true_pa = df_pa["y_true"].values
        y_pred_pa = df_pa["y_pred"].values

        y_true_nonpa = df_nonpa["y_true"].values
        y_pred_nonpa = df_nonpa["y_pred"].values

                                    
              
                                    
        r_pa, rmse_pa = calc_metrics(y_true_pa, y_pred_pa)
        r_nonpa, rmse_nonpa = calc_metrics(y_true_nonpa, y_pred_nonpa)

        print(f"PAs: r = {r_pa:.2f}, RMSE = {rmse_pa:.2f}")
        print(f"Non-PAs: r = {r_nonpa:.2f}, RMSE = {rmse_nonpa:.2f}")

                                    
             
                                    
        fig, ax = plt.subplots(figsize=(2.25, 2.15))
        plt.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.13)
                                    
            
                                    
        ax.scatter(
            y_true_pa,
            y_pred_pa,
            s=45,
            color=color_pa,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.6,
            zorder=3,
            label="PAs"
        )

        ax.scatter(
            y_true_nonpa,
            y_pred_nonpa,
            s=45,
            color=color_nonpa,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.6,
            zorder=3,
            label="Non-PAs"
        )
                                    
               
                                    
        ax.plot(
            [-0.5, 1.3],
            [-0.5, 1.3],
            linestyle="--",
            color="#D62728",
            linewidth=1,
            zorder=1
        )

                                    
             
                                    
        ax.set_xlabel("Observed dNBR", fontsize=8, fontweight="normal")
        ax.set_ylabel("Predicted dNBR", fontsize=8, fontweight="normal")

        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.125, 0.5)

        ax.set_xlim(-0.5, 1.3)
        ax.set_ylim(-0.5, 1.3)

        ax.set_xticks([-0.4, 0, 0.4, 0.8, 1.2])
        ax.set_yticks([-0.4, 0, 0.4, 0.8, 1.2])

        ax.set_aspect('equal', adjustable='box')

                                    
                         
                                    
        text_str = (
            f"PAs: r = {r_pa:.2f}, RMSE = {rmse_pa:.2f}\n"
            f"Non-PAs: r = {r_nonpa:.2f}, RMSE = {rmse_nonpa:.2f}"
        )

        ax.text(
            0.98, 0.02,                         
            text_str,
            transform=ax.transAxes,
            fontsize=7.5,
            ha='right',
            va='bottom'
        )

                                    
            
                                    
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper left",
            ncol=2,
            columnspacing=0.6,
            handletextpad=0.1,
            borderpad=0.2
        )

                
        ax.tick_params(
            direction="in",
            length=3,
            width=0.8,
            labelsize=8
        )

            
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            color="0.85",
            zorder=0
        )

            
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

                                    
            
                                    
        save_path = os.path.join(root_path, f"Biome_{biome}", "observed_vs_predicted.png")

        plt.savefig(
            save_path,
            dpi=600
        )

        plt.close()

        print(f"✅ Saved: {save_path}")


if __name__ == '__main__':
    main()
