import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

# File for text report generation
filename = ROOT / "result_report.txt"


sns.set_style("darkgrid")


def _expand_distance_columns(results_df):
    distances = results_df["distance"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    )

    max_length = distances.map(len).max()
    distance_columns = [f"dist_{idx + 1}" for idx in range(max_length)]

    dist_expanded = pd.DataFrame(
        distances.tolist(),
        columns=distance_columns,
        index=results_df.index,
    )

    results_df = pd.concat([results_df, dist_expanded], axis=1)
    results_df["Distance (px)"] = results_df[distance_columns].mean(axis=1)

    return results_df

def plot_policy_comparison_ensemble(results_df):
    """
    Plots a comparison of different policies based on the results DataFrame.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the results of different policies.
    """
    results_df_ensemble = results_df[results_df['model_name'] == 'ensemble'].copy()  # Exclude the oracle policy for comparison
    rmse_normalized = results_df_ensemble.groupby(["experiment_id", "policy", "model_name"])["rmse"].transform(lambda x: x / x.iloc[0])  # Normalize RMSE by distance 
    
    results_df_ensemble["RMSE_normalized"] = rmse_normalized
    
    results_df_ensemble = _expand_distance_columns(results_df_ensemble.copy())

    # Create a bar plot comparing the policies
    plt.figure(figsize=(7, 4))
    
    # Rename columns to beautify
    results_df_ensemble = results_df_ensemble.rename(columns={"policy": "Policy", "RMSE_normalized": "Normalized RMSE"})
    
    # Rename the policies for better visualization
    policy_mapping = {
        "epsilon_greedy": r"$\epsilon$-greedy",
        "orienteering": "Receeding Horizon Orienteering",
        "myopic_greedy": "Value Greedy",
        "mcts": "MCTS",
        "uncertainty_greedy": "Uncertainty Greedy",
    }
    results_df_ensemble["Policy"] = results_df_ensemble["Policy"].replace(policy_mapping)

    sns.lineplot(x='Distance (px)', y='Normalized RMSE', hue="Policy", data=results_df_ensemble, markers=True, dashes=False, style="Policy", palette="tab10")
    plt.xlim(results_df_ensemble["Distance (px)"].min(), 300)  # Set x-axis limits based on the maximum distance
    
    plt.legend(loc="upper right", ncols=2)

    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig(ROOT / "policy_comparison_ensemble.png", dpi=300)  # Save the figure with high resolution
    # Save also as svg for better quality in publications
    plt.savefig(ROOT / "policy_comparison_ensemble.svg")

    # Write the results to a text file for reporting
    with open(filename, "w") as f:
        
        print("-" * 50, file=f)
        print("Policy Comparison Report:\n", file=f)

        print("The model is ENSEMBLE for this comparison." \
        "The RMSE is normalized by the error in the first step (distance) " \
        "to account for the varying difficulty of the experiments.", file=f)

        for policy in results_df_ensemble["Policy"].unique():
            policy_df = results_df_ensemble[results_df_ensemble["Policy"] == policy]
            avg_rmse = policy_df.groupby(["experiment_id"])["Normalized RMSE"].last().mean()
            std_rmse = policy_df.groupby(["experiment_id"])["Normalized RMSE"].last().std()
            f.write(f"Policy: {policy}, Average RMSE (normalized): {avg_rmse:.4f}, Std Dev: {std_rmse:.4f}\n")
    
def plot_boxplot_models(results_df):
    """
    Plots a boxplot comparing the RMSE of different models.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the results of different models.
    """
    
    df = results_df.copy()
        
    df["Normalized RMSE"] = df.groupby(["experiment_id", "policy", "model_name"])["rmse"].transform(lambda x: x / x.iloc[0])  # Normalize RMSE by distance
    
    # Select the last entry for each experiment_id and policy and modeºl to avoid duplicates
    df = df.drop_duplicates(subset=['experiment_id', 'policy', 'model_name'], keep='last')
    
    df = df.rename(columns={"policy": "Policy", "distance_bin": "Distance (px)", "model_name": "Model"})

    
    policy_mapping = {
        "epsilon_greedy": r"$\epsilon$-greedy",
        "orienteering": "Receeding Horizon Orienteering",
        "myopic_greedy": "Value Greedy",
        "mcts": "MCTS",
        "uncertainty_greedy": "Uncertainty Greedy",
    }
    df["Policy"] = df["Policy"].replace(policy_mapping)
    
    model_mapping = {
        "ensemble": "Deep Ensemble",
        "gaussian_process": "Gaussian Process",
    }
    df["Model"] = df["Model"].replace(model_mapping)
     

    plt.figure(figsize=(7, 4))
    
    
    sns.boxplot(x='Model', y='Normalized RMSE', hue='Policy', data=df, palette="tab10", showfliers=False)
    
    plt.xlabel("Model")
    plt.ylabel("Normalized RMSE")
    plt.title("Comparison of Normalized RMSE across Models")
    
    plt.tight_layout()
    
    # Show the plot
    # plt.show()

    plt.savefig(ROOT / "boxplot_models.png", dpi=300)  # Save the figure with high resolution
    # Save also as svg for better quality in publications
    plt.savefig(ROOT / "boxplot_models.svg")

    # Write the results to a text file for reporting
    with open(filename, "a") as f:
        print("\n" + "-" * 50, file=f)
        print("Model Comparison Report:\n", file=f)
        print("Comparison between different models based on their normalized RMSE.", file=f)

        for model in df["Model"].unique():
            model_df = df[df["Model"] == model]
            # Take the last entry for each experiment_id and policy to avoid duplicates
            avg_last_rmse = model_df.groupby(["experiment_id"])["Normalized RMSE"].last().median()
            std_last_rmse = model_df.groupby(["experiment_id"])["Normalized RMSE"].last().std()
            f.write(f"Model: {model}, Average RMSE (normalized): {avg_last_rmse:.4f}, Std Dev: {std_last_rmse:.4f}\n")
    
def plot_iou_lineplot(results_df):
    """
    Plots a line plot comparing the IoU of different policies.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the results of different policies.
    """
    
    results_df = results_df[results_df['model_name'] == 'ensemble']  # Exclude the oracle policy for comparison
    results_df = _expand_distance_columns(results_df.copy())

    # Create a line plot comparing the policies
    plt.figure(figsize=(7, 4))
    
    # Rename columns to beautify
    results_df = results_df.rename(columns={"policy": "Policy", "iou": "IoU"})
    
    # Rename the policies for better visualization
    policy_mapping = {
        "epsilon_greedy": r"$\epsilon$-greedy",
        "orienteering": "Receeding Horizon Orienteering",
        "myopic_greedy": "Value Greedy",
        "mcts": "MCTS",
        "uncertainty_greedy": "Uncertainty Greedy",
    }
    results_df["Policy"] = results_df["Policy"].replace(policy_mapping)

    sns.lineplot(x='Distance (px)', y='IoU', hue="Policy", data=results_df, markers=True, dashes=False, style="Policy", palette="tab10")
    plt.xlim(results_df["Distance (px)"].min(), 300)  # Set x-axis limits based on the maximum distance
    
    plt.legend(loc="lower right", ncols=2)

    plt.tight_layout()

    # Show the plot
    #plt.show()
    plt.savefig(ROOT / "iou_comparison.png", dpi=300)  # Save the figure with high resolution
    # Save also as svg for better quality in publications
    plt.savefig(ROOT / "iou_comparison.svg")

    # Write the results to a text file for reporting
    with open(filename, "a") as f:
        
        print("\n" + "-" * 50, file=f)
        print("IoU Comparison Report:\n", file=f)
        print("Comparison of IoU across different policies using the ENSEMBLE model.", file=f)

        for policy in results_df["Policy"].unique():
            policy_df = results_df[results_df["Policy"] == policy]
            avg_last_iou = policy_df.groupby(["experiment_id"])["IoU"].last().mean()
            std_last_iou = policy_df.groupby(["experiment_id"])["IoU"].last().std()
            f.write(f"Policy: {policy}, Average IoU: {avg_last_iou:.4f}, Std Dev: {std_last_iou:.4f}\n")
    
def plot_example_trajectory(results_df, maps, experiment_id):

    results_df = results_df[results_df['model_name'] == 'ensemble']  # Exclude the oracle policy for comparison
    
    policy_mapping = {
        "epsilon_greedy": r"$\epsilon$-greedy",
        "orienteering": "Receeding Horizon\nOrienteering",
        "myopic_greedy": "Value Greedy",
        "mcts": "MCTS",
        "uncertainty_greedy": "Uncertainty Greedy",
    }
    results_df["Policy"] = results_df["policy"].replace(policy_mapping)

    
    example_df = results_df.copy()
    
    example_df['experiment_id'] = results_df['experiment_id'].apply(lambda x: x % 100)  # Modulo to wrap around experiment IDs if they exceed 200
    example_df = example_df[example_df['experiment_id'] == experiment_id]

    
    # Drop those rows with distance greater than 300 for better visualization
    
    n_subplots = len(example_df['Policy'].unique())
    print(n_subplots)
    
    fig, axs = plt.subplots(1, n_subplots, figsize=(13, 4))
    
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]  # Ensure axs is always a list of axes

    
    for idx, (policy, policy_df) in enumerate(example_df.groupby('Policy')):
        
        ax = axs[idx] if idx < len(axs) else axs[0]  # Use the corresponding subplot or fallback to the first one
        
        ax.set_aspect('equal')  # Set equal aspect ratio for correct spatial representation
        ax.grid(False)  # Disable grid for better visualization
        # Extract X, Y coordinates from the trajectory column for the given experiment_id and policy
        x = policy_df['pos_x'].values
        y = policy_df['pos_y'].values
        
        trajectory = np.asarray([x, y], dtype=int).T  # Convert X and Y columns to a numpy array of shape (n_steps, 2)
        
    
        ax.imshow(maps[experiment_id - 1], cmap='viridis', interpolation='bicubic')  # Display the map as a background
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', marker='.', label=policy, alpha=0.5)  # Plot the trajectory on top of the map
        
        ax.set_title(f"{policy}")
        
        
        ax.set_xlabel("X")
        if idx == 0:  # Only set y-label for the first subplot to avoid clutter
            
            ax.set_ylabel("Y")
        else:
            ax.set_yticks([])  # Hide y-ticks for other subplots for better visualization
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(ROOT / f"example_trajectory_experiment_{experiment_id}.png", dpi=300)  # Save the figure with high resolution
    # Save also as svg for better quality in publications
    plt.savefig(ROOT / f"example_trajectory_experiment_{experiment_id}.svg")
    
def heat_map_RMSE_policy_model(results_df):
    """
    Plots a heatmap comparing the RMSE of different policies and models.

    Parameters:
    results_df (pd.DataFrame): A DataFrame containing the results of different policies and models.
    """
    
    df = results_df.copy()

    df["RMSE_normalized"] = df.groupby(["experiment_id", "policy"])["rmse"].transform(lambda x: x / x.iloc[0])  # Normalize RMSE by distance
    
    # Select the last entry for each experiment_id and policy and model to avoid duplicates
    df = df.drop_duplicates(subset=['experiment_id', 'policy', 'model_name'], keep='last')
    
    df = df.rename(columns={"policy": "Policy", "distance_bin": "Distance (px)", "RMSE_normalized": "Normalized RMSE", "model_name": "Model"})

    
    policy_mapping = {
        "epsilon_greedy": r"$\epsilon$-greedy",
        "orienteering": "Receeding Horizon Orienteering",
        "myopic_greedy": "Value Greedy",
        "mcts": "MCTS",
        "uncertainty_greedy": "Uncertainty Greedy",
    }
    df["Policy"] = df["Policy"].replace(policy_mapping)
    
    model_mapping = {
        "ensemble": "Deep Ensemble",
        "gaussian_process": "Gaussian Process",
    }
    df["Model"] = df["Model"].replace(model_mapping)
     
    pivot_table = df.pivot_table(index='Model', columns='Policy', values='Normalized RMSE', aggfunc='mean')
    
    # Order by the average RMSE across policies for better visualization
    pivot_table = pivot_table.loc[pivot_table.mean(axis=1).sort_values().index]
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    
    plt.title("Heatmap of RMSE by Policy and Model")
    
    plt.tight_layout()
    
    # Show the plot
    # plt.show()

    plt.savefig(ROOT / "heatmap_rmse_policy_model.png", dpi=300)  # Save the figure with high resolution
    # Save also as svg for better quality in publications
    plt.savefig(ROOT / "heatmap_rmse_policy_model.svg")

    print("\n" + "-" * 50, file=open(filename, "a"))
    print("Heatmap of RMSE by Policy and Model Report:\n", file=open(filename, "a"))
    print("This heatmap shows the average RMSE for each combination of policy and model. The RMSE is normalized by the error in the first step (distance) to account for the varying difficulty of the experiments.", file=open(filename, "a"))
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        avg_rmse = model_df["rmse"].mean()

        with open(filename, "a") as f:
            f.write(f"Model: {model}, Average RMSE (normalized): {avg_rmse:.4f}, Std Dev: {model_df['rmse'].std():.4f}\n")
    



    
if __name__ == "__main__":
    
    # Load the results from a CSV file
    results_df = pd.read_csv(ROOT / "experimentstest.csv")

    # Drop rows with distance greater than 300 for better visualization
    # results_df = results_df[results_df['distance'] <= 305]

    # Plot the policy comparison
    plot_policy_comparison_ensemble(results_df)
    
    # Plot the boxplot of models
    plot_boxplot_models(results_df)
    
    # Plot the IoU line plot
    plot_iou_lineplot(results_df)
    
    # Plot an example trajectory for a specific experiment ID
    example_experiment_id = 2  # Change this to the desired experiment ID
    # Load the maps for the example trajectory plot
    maps = np.load(ROOT.parent.parent / "dataset" / "dataset_POINTWISE.npz")
    results_df = pd.read_csv(ROOT / "experimentstest.csv")
    plot_example_trajectory(results_df, maps, example_experiment_id)

    # Plot the heatmap of RMSE by policy and model
    heat_map_RMSE_policy_model(results_df)
    