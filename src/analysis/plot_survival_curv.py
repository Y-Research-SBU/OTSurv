import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_km(high_risk_times, high_risk_events, low_risk_times, low_risk_events, save_path):
    plt.figure(figsize=(8, 6), dpi=150)  # High resolution
    kmf = KaplanMeierFitter()

    # Define custom colors and line styles
    colors = {"High Risk": "#E74C3C", "Low Risk": "#3498DB"}
    linestyles = {"High Risk": "-", "Low Risk": "--"}

    # High-risk group
    kmf.fit(high_risk_times, event_observed=high_risk_events, label="High Risk")
    kmf.plot_survival_function(
        ci_show=True,  # Show confidence interval
        color=colors["High Risk"],
        linestyle=linestyles["High Risk"],
        linewidth=2.5
    )

    # Low-risk group
    kmf.fit(low_risk_times, event_observed=low_risk_events, label="Low Risk")
    kmf.plot_survival_function(
        ci_show=True, 
        color=colors["Low Risk"],
        linestyle=linestyles["Low Risk"],
        linewidth=2.5
    )

    # Perform log-rank test to get p-value
    result = logrank_test(
        high_risk_times, low_risk_times, 
        event_observed_A=high_risk_events, 
        event_observed_B=low_risk_events
    )
    p_value = result.p_value

    # Format p-value in scientific notation with 2 decimal places
    p_text = f"Log-rank p-value: {p_value:.2e}"

    # # Title and labels with larger fonts
    plt.title(p_text, fontsize=20)
    plt.xlabel("Time (Days)", fontsize=20)
    plt.ylabel("Survival Probability", fontsize=20)

    # Improve grid styling
    plt.grid(True, linestyle="--", alpha=0.6)

    # Customize legend
    plt.legend(fontsize=20, loc="best", frameon=False)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")


def cal_score(all_risk_scores, all_censorships, all_event_times, save_path):
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)

    # use median of risk scores to group
    # threshold = np.median(all_risk_scores)
    threshold_high = np.percentile(all_risk_scores, 50)
    threshold_low = np.percentile(all_risk_scores, 50)
    high_risk_idx = all_risk_scores > threshold_high  # high risk group
    low_risk_idx = all_risk_scores <= threshold_low  # low risk group

    # extract data from high risk and low risk groups
    high_risk_times = all_event_times[high_risk_idx]
    high_risk_events = all_censorships[high_risk_idx]

    low_risk_times = all_event_times[low_risk_idx]
    low_risk_events = all_censorships[low_risk_idx]

    high_risk_events = 1 - high_risk_events  # reverse censoring mark
    low_risk_events = 1 - low_risk_events

    # calculate log-rank test
    result = logrank_test(
        high_risk_times, low_risk_times,
        event_observed_A=high_risk_events,
        event_observed_B=low_risk_events
    )
    plot_km(high_risk_times, high_risk_events, low_risk_times, low_risk_events, save_path)
    return result



dataset_list = ["BRCA", "BLCA", "LUAD", "STAD", "COADREAD", "KIRC"]
fig_save_dir = "../result/visulization"
testfile_dir = "../result/exp_otsurv_test"
for dataset in dataset_list:

    save_path = os.path.join(fig_save_dir, dataset + "_km.png")

    print(dataset + " ============")
    all_risk_scores_list = []
    all_censorships_list = []
    all_event_times_list = []
    for fold in range(5):
        file_path = glob(os.path.join(testfile_dir, dataset + "_survival/k=" + str(fold) + "/*/*/all_dumps.h5"))
        if len(file_path) != 1:
            print(f"No file or multiple files found for {dataset} fold {fold}")
            continue
        file_path = file_path[0]

        with open(file_path,'rb') as f:
            file = pickle.load(f)

        all_risk_scores = file["test"]['all_risk_scores'].tolist()
        all_censorships = file["test"]['all_censorships'].tolist()
        all_event_times = file["test"]['all_event_times'].tolist()

        all_risk_scores_list += all_risk_scores
        all_censorships_list += all_censorships
        all_event_times_list += all_event_times

        # result = cal_score(all_risk_scores, all_censorships, all_event_times, save_path)
        # # print result
        # print(name + " " + str(fold) + f" Log-rank p-value: {result.p_value}")
    
    result = cal_score(all_risk_scores_list, all_censorships_list, all_event_times_list, save_path)

    print(result.p_value)