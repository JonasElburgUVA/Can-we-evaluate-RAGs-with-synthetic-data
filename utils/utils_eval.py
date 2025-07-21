"""
These functions are used in the TODO notebooks to construct the tables and figures presented in the report.
"""
from collections import defaultdict
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, linregress
import matplotlib.cm as cm
import numpy as np
from . import get_metric_group
from .utils import get_evaluator_models, per_model_costs
from ragas.cost import get_token_usage_for_openai
import yaml
import itertools
root = os.environ['PROJECT_ROOT']

def eval_dataset(ds, config, overwrite=False, upload=False):
    eval_llm, eval_embedding = get_evaluator_models(
    llm_deployment=config["eval_llm_deployment"],
    embedding_deployment=config["eval_embedding_deployment"]
    )
    metric_group = get_metric_group(
        name=config["metric_group"]
    )
    results = {}
    simple_results = {}
    for buddy in config["buddies"]:
        try:
            results[buddy] = ds.evaluate(
                buddy_name=buddy,
                metrics=metric_group,
                llm=eval_llm,
                embeddings=eval_embedding,
                token_usage_parser=get_token_usage_for_openai,
                overwrite=overwrite
            )

            if len(results[buddy].cost_cb.usage_data) > 0:
                # Calculate rough cost of input and output tokens for the used evaluation LLMs (euros)
                print(f"Cost: {results[buddy].total_cost(per_model_costs=per_model_costs)}")
            simple_results[buddy] = results[buddy]._repr_dict
            if upload:
                link = results[buddy].upload()
                simple_results[buddy]["link"] = link
        except FileExistsError:
            print("Results are already calculated. If you want to change metrics, set overwrite=True.")
            results[buddy], simple_results[buddy] = ds.load_results(buddy)

    return results, simple_results


def load_results(output_path:str, data_types:list, buddies:list):
    """Loads the JSON results into dictionaries"""
    all_results = defaultdict(lambda: defaultdict())
    all_results_simple = defaultdict(lambda: defaultdict())
    for data_type in data_types:
        for buddy in buddies:
            result_dir = os.path.join(output_path, data_type, buddy, "results")
            with open(result_dir + "\\results.json", "r", encoding="utf8") as f:
                results = json.load(f)
            with open(result_dir + "/simple_results.json", "r", encoding="utf8") as f:
                simple_results = json.load(f)
            all_results[data_type][buddy] = dict(results)
            all_results_simple[data_type][buddy] = dict(simple_results)

    return all_results, all_results_simple


def combine_results(df_human, df_gen):
    # Concatenating with keys for differentiation
    df_comb = pd.concat([df_human, df_gen], keys=["Human", "Generated"])

    # Stacking to create the hierarchical structure
    df_comb = df_comb.stack().apply(pd.Series).stack().reset_index()

    # Renaming columns properly
    df_comb.columns = ['Dataset',  'Metric', 'Model', 'Index', 'Value']

    # Setting hierarchical index
    df_comb = df_comb.set_index(['Dataset', 'Metric', 'Model', 'Index'])

    return df_comb

def plot_ecdf_metric(metric, df, dataset, output_path, save=True,dpi=1000):
    # Extract Human dataset for the metric
    df_human_metric = df.xs(("Human", metric), level=("Dataset", "Metric")).reset_index()

    # Extract Generated dataset for the metric
    df_gen_metric = df.xs(("Synthetic", metric), level=("Dataset", "Metric")).reset_index()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, dpi=dpi)

    # Plot for Human dataset
    sns.ecdfplot(data=df_human_metric, x="Value", hue="Model", complementary=True, ax=axes[0], legend=False)
    axes[0].set_title("ECDF of Semantic Similarity (Human)")
    axes[0].set_xlabel("Metric Score")
    axes[0].set_ylabel("Proportion of Samples")

    # Plot for Generated dataset
    sns.ecdfplot(data=df_gen_metric, x="Value", hue="Model", complementary=True, ax=axes[1])
    axes[1].set_title("ECDF of Semantic Similarity (Synthetic)")
    axes[1].set_xlabel("Metric Score")

    # Adjust layout and show plot
    plt.tight_layout()

    if save:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(output_path + f"/{dataset}_{metric}.png", format="png", dpi=dpi)

    return fig

def get_kendall_stats(df):
    # Step 2: Reshape so that each metric remains a row, models are columns, and datasets form separate columns
    df_reset = df.reset_index()
    df_pivot = df_reset.pivot(index=['Metric', 'Model'], columns='Dataset', values='Rank')

    # Step 3: Compute Kendall’s Tau for each metric
    kendall_results = {}
    for metric in df_pivot.index.get_level_values('Metric').unique():
        metric_data = df_pivot.loc[metric]
        
        # Ensure there are no NaNs before computing Kendall's Tau
        metric_data.dropna(inplace=True)
        
        tau, p_value = kendalltau(metric_data['Human'], metric_data['Synthetic'])
        kendall_results[metric] = {'Kendall Tau': tau, 'P-value': p_value}

    # Convert results to DataFrame and display
    kendall_df = pd.DataFrame.from_dict(kendall_results, orient='index')

    return kendall_df


def compute_percentual_change(group, baseline_model):
    # Compute percentual change compared to the baseline model in each (Dataset, Metric) group
    baseline_value = group.xs(baseline_model, level="Model")["Value"]
    # baseline_value = group.loc[baseline_model, 'Value'] if baseline_model in group.index else None
    if baseline_value.values not in (None, 0):
        group['Percentual Change'] = (group['Value'] - baseline_value) / abs(baseline_value) * 100
    else:
        group['Percentual Change'] = None  # If baseline model is missing
    return group

def plot_percentual_change(df, models, output_path, dpi=1000, save=False, log_scale=True):

    fig, axs = plt.subplots(1,len(models),sharex=False,sharey=False, figsize=(6*len(models),5), dpi=dpi)
    # metrics = ...
    for i,model in enumerate(models):
        if len(models) == 1:
            ax = axs
        else:
            ax = axs[i]
        sns.barplot(data=df.xs(model, level="Model"), x='Metric', y='Percentual Change', hue='Dataset', dodge=True, ax=ax, width=0.5)
        if i == 0:
            ax.set_ylabel("Percentual Change (%)")
        else:
            ax.set_ylabel("")
        ax.axhline(0, color='black', linewidth=1, linestyle='dashed')  # Baseline at 0%
        ax.set_xlabel("Metric")
        ax.tick_params(rotation=45)
        if i == (len(models)-1):
            ax.legend(title="Benchmark")
        else:
            ax.legend().remove()
        ax.set_title(model, pad=10)
        if log_scale:
            ax.set_yscale("symlog")

    fig.suptitle("Percentage Change by Metric and Benchmark of alternative models \n compared to the baseline (GPT-4o-mini)", fontsize=18)
    plt.subplots_adjust(top=0.78)

    if save:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(output_path+"/percentual_change.png", dpi=dpi, bbox_inches='tight', format='png')
    return fig


def plot_answer_length_per_metric(df, metrics_to_plot, output_path, dpi=1000, regression_line=True):
    # Extract only the "response" metric while keeping all index levels
    answer_df = df.xs("response", level="Metric", drop_level=False).copy()

    # Compute string lengths
    answer_length_df = answer_df.copy()
    answer_length_df["Value"] = answer_length_df["Value"].apply(lambda x: len(x.split())) # Ensure string conversion)

    # Replace "response" with "Answer Length" in the Metric level while keeping other index levels
    answer_length_df.index = answer_length_df.index.set_levels(
        answer_length_df.index.levels[1].tolist() + ["Answer Length"], level="Metric"
    )
    answer_length_df = answer_length_df.rename(index={"response": "Answer Length"})

    # Concatenate with the original DataFrame
    df_comb = pd.concat([df, answer_length_df]).sort_index()

    answer_length_df = df_comb.xs("Answer Length", level="Metric")

    metrics_df = df_comb.loc[df_comb.index.get_level_values("Metric").isin(metrics_to_plot)]
    # # Pivot to align "Answer Length" with the selected metrics
    metrics_df = metrics_df.unstack("Metric")  # Convert multi-index to columns
    merged_df = metrics_df["Value"].join(answer_length_df, rsuffix="_AnswerLength")

    
    # Get unique models and assign colors
    unique_models = merged_df.index.get_level_values("Model").unique()
    model_colors = {model: cm.tab10(i) for i, model in enumerate(unique_models)}

    # Ensure metrics_to_plot exists and contains valid metric names
    if len(metrics_to_plot) == 1:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        axes = [ax]  # Convert to list for consistency
    else:
        fig, axes = plt.subplots(nrows=len(metrics_to_plot)//2, ncols=2, figsize=(10, 4 * len(metrics_to_plot)//2), dpi=1000, sharex=True, sharey=True)

    # Iterate over metrics and create subplots
    for i, (ax, metric) in enumerate(zip(axes.flatten(), metrics_to_plot)):
        x = merged_df["Value"].values  # Ensure it's a NumPy array
        y = merged_df[metric].values  # Ensure it's a NumPy array

        # Check if x and y have enough values for regression
        if len(x) > 1 and len(y) > 1:
            models = merged_df.index.get_level_values(level="Model")
            for model in unique_models:
                mask = models == model  # Filter data for this model
                ax.scatter(x[mask], y[mask], color=model_colors[model], alpha=0.5, label=model)
            
            # Perform linear regression to get best fit line
            if regression_line:
                for model in unique_models:
                    mask = models == model
                    m, b, r, p, _ = linregress(x[mask], y[mask])
                    # Generate regression line points
                    x2 = np.linspace(x.min(), x.max(), 100)  # Ensure valid range
                    y2 = m * x2 + b
                    ax.plot(x2, y2, color=model_colors[model], linestyle='--')

        else:
            ax.text(0.5, 0.5, "Not enough data for regression", ha='center', va='center', fontsize=12)

        # Labels and title
        if i >= 2:
            ax.set_xlabel("Answer Length")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        # ax.set_title(f"Answer Length vs {metric}")
        if ax == axes.flatten()[0]: 
            ax.legend()

    plt.suptitle("Metric score by answer length", fontsize=20)
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_path}/answer_length.png", dpi=dpi, format="png")
    return fig


def sorted_buddy_bars(data, palette, **kwargs):
    """
    Draws bars for each Datatype in 'data', sorted by 'mean' descending,
    and color-coded by Buddy. One 'cluster' of bars per Datatype.
    """
    ax = plt.gca()

    # 4a. Identify the datatypes in this facet & fix their order (if you want a custom Datatype order)
    datatypes = data['Datatype'].unique()
    datatypes = sorted(datatypes)  # or use any custom ordering

    # 4b. We will place each Datatype cluster at x_positions[idx].
    x_positions = np.arange(len(datatypes))

    # 4c. For each Datatype: sort the subset by 'mean' desc, then plot bars side-by-side in that cluster
    for idx, dt in enumerate(datatypes):
        subset = data[data['Datatype'] == dt].copy()
        # Sort Buddies by descending mean for *this* Datatype
        subset.sort_values(by="mean", ascending=True, inplace=True)

        # Number of buddies in *this* Datatype cluster
        nbuddies = len(subset)
        # The total cluster width we want on the x-axis
        cluster_width = 0.8  
        # How wide each bar will be
        bar_width = cluster_width / nbuddies
        
        # Plot each Buddy’s bar, in sorted order
        for i, row in enumerate(subset.itertuples()):
            buddy = row.Buddy
            mean_ = row.mean
            var_ = row.var
            color = palette[buddy]  # same color for the same Buddy across the entire grid
            
            # x-position for this bar = center of the cluster +/- offset
            left_edge = x_positions[idx] - cluster_width/2
            x_pos = left_edge + i * bar_width + bar_width/2
            
            ax.bar(
                x_pos, 
                mean_,
                width=bar_width, 
                color=color, 
                edgecolor='black',
                yerr=var_,
                capsize=3
            )
    
    # 4d. Label the cluster positions on the x-axis with Datatype names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datatypes, rotation=0, ha='center')

    # 4e. Optional: add vertical grid lines or otherwise style
    ax.grid(axis='y', alpha=0.3)

def plot_meta_rankings(filtered_df, datasets, row_order=None, legend_offset=-.05):
        # -------------------------------------------------------------
    # 2. Create a color palette dict: Buddy -> color
    # -------------------------------------------------------------
    unique_buddies = list(filtered_df['Buddy'].unique())
    # Move 'Baseline' to the front if it exists
    for buddy in unique_buddies:
        if 'Baseline' in buddy:
            unique_buddies.remove(buddy)
            unique_buddies.insert(0, buddy)
    palette = dict(zip(unique_buddies, sns.color_palette('bright', len(unique_buddies))))

    # -------------------------------------------------------------
    # 3. Build a FacetGrid (row=Metric, col=Dataset for example)
    # -------------------------------------------------------------
    g = sns.FacetGrid(
        filtered_df, 
        row="Metric", 
        col="Dataset", 
        sharex=True, 
        sharey=False,   # or True, your choice
        margin_titles=True,
        col_order=datasets,
        row_order = row_order,
        height=3,
        aspect=1.5
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=21)
    # -------------------------------------------------------------
    # 5. Map the custom function onto each Facet
    # -------------------------------------------------------------
    g.map_dataframe(sorted_buddy_bars, palette=palette)

    # -------------------------------------------------------------
    # 6. (Optional) Build a custom legend for the Buddies
    # -------------------------------------------------------------
    # We'll build a legend manually so each Buddy is labeled with correct color.
    # Because we called ax.bar() manually, Seaborn won't auto-build the legend for us.
    axes = g.axes.flatten()  # all axes in the grid
    # Tweak font sizes
    for ax in axes:
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=17)
    handles = []
    labels = []
    # Loop through each Buddy -> color to create handles
    for buddy, color in palette.items():
        handles.append(
            plt.Line2D(
                [0], [0], 
                marker='s',
                color=color,
                label=buddy,
                markersize=8,
                linewidth=0
            )
        )
        labels.append(buddy)

    # Add that legend to, say, the top-right subplot
    fig = g.figure  # the matplotlib Figure object inside your FacetGrid
    fig.legend(
        handles,
        labels,
        title="Buddy",
        loc="lower center",       # place it at the bottom
        bbox_to_anchor=(0.5, legend_offset),  # adjust downward (negative y) as needed
        ncol=len(labels),         # spread out horizontally, or pick a custom value
        title_fontsize=24,
        fontsize=20
    )

    # fig.suptitle(f"Mean metrics scores", fontsize=20)
    # -------------------------------------------------------------
    # 7. Show the figure
    # -------------------------------------------------------------
    plt.tight_layout()
    plt.show()
    return fig

def plot_meta_descriptives(datasets, datatypes, experiment_name, datatype_path_names):
    df_descriptives = pd.DataFrame(columns=['Dataset', 'Datatype', 'Question length mean', 'Question length sd', "Answer length mean", "Answer length sd"])

    # Build the DataFrame
    for (ds, dt) in itertools.product(datasets, datatypes):
        path = f'{root}/output/{ds}/{experiment_name}/{datatype_path_names[dt]}/question_descriptives.json'
        if not os.path.exists(path):
            raise NameError(f"Path {path} does not exist!")
            continue
        with open(path, "rb") as f:
            descriptives = json.load(f)

        df_descriptives.loc[len(df_descriptives)] = (
            ds, dt, descriptives["Mean question length"], descriptives["SD question length"], descriptives["Mean answer length"], descriptives["SD answer length"]
        )
    # Shared bar setup
    x = np.arange(len(datatypes))  # positions for each datatype
    width = 0.8 / len(datasets)    # width of each bar

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=1000, sharey=False)

    # ---- QUESTION LENGTH ----
    for i, ds in enumerate(datasets):
        subset = df_descriptives[df_descriptives['Dataset'] == ds]
        means = subset.set_index('Datatype').loc[datatypes]['Question length mean'].astype(float)
        sds = subset.set_index('Datatype').loc[datatypes]['Question length sd'].astype(float)
        bar_positions = x + i * width

        ax1.bar(bar_positions, means, width=width, label=ds, yerr=sds, capsize=4)

    ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax1.set_xticklabels(datatypes, size=18)
    ax1.set_ylabel('Length (words)', size=18, labelpad=10)
    ax1.set_title('Questions', size=22)
    # ax1.legend(title='Dataset')

    # ---- ANSWER LENGTH ----
    for i, ds in enumerate(datasets):
        subset = df_descriptives[df_descriptives['Dataset'] == ds]
        means = subset.set_index('Datatype').loc[datatypes]['Answer length mean'].astype(float)
        sds = subset.set_index('Datatype').loc[datatypes]['Answer length sd'].astype(float)
        bar_positions = x + i * width

        ax2.bar(bar_positions, means, width=width, label=ds, yerr=sds, capsize=4)

    ax2.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax2.set_xticklabels(datatypes, size=18)
    ax2.set_title('Reference Answers', size=22)
    ax2.legend(title="Dataset", fontsize=18)

    plt.tight_layout(pad=0.1)

    return fig

def get_combined_df(config_names):
    configs={}
    dataset_names = {}
    data_types = {}
    output_paths = {}
    buddies = {}
    results = {}
    simple_results = {}
    baseline = {}
    for c in config_names:
        with open(os.path.join(root, "configs", "experiments", f"{c}.yaml"), "r") as f:
            config = yaml.safe_load(f)
        configs[c] = config
        data_types[c] = ["Human", config["generated_dataset_name"]]
        output_paths[c] = os.path.join(root, "output", config["dataset_name"], config["experiment_name"])
        dataset_names[c] = config['dataset_name']
        buddies[c] = config['buddies']
        baseline[c] = config['baseline']

        results[dataset_names[c]], simple_results[dataset_names[c]] = load_results(
            output_path=output_paths[c],
            data_types=data_types[c],
            buddies=buddies[c]
        )
        # Add FN ratio
        for dt, bd in itertools.product(data_types[c], buddies[c]):
            results[dataset_names[c]][dt][bd]['false_negatives'] = simple_results[dataset_names[c]][dt][bd]['false_negatives']
        
    df = pd.DataFrame.from_dict(results)
    df = df.stack().apply(pd.Series).stack().apply(pd.Series).stack().apply(pd.Series).stack().reset_index()
    df.columns = ['Datatype','Dataset', 'Buddy', 'Metric', 'Index', 'Value']
    df = df.set_index(['Dataset','Datatype', 'Buddy', 'Metric', 'Index'])
    df.head()
    return df

