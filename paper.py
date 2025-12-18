#%%
# Extracts the data from log files and analyses them
import log_parser
import get_analysis

log_parser.main()
man1, man2 = get_analysis.main()

#%%
# loads the analysis dataa
import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('expand_frame_repr', False)

zuordnung = pd.read_csv('data/zuordnung.csv')
zuordnung.rename(columns=str.lower, inplace=True)  # Converts 'Date' → 'date'

summary = pd.read_csv('data/summary.csv')
summary.rename(columns=str.lower, inplace=True)

analysis = pd.read_csv('data/analysis.csv')

all = pd.merge(analysis, zuordnung, on='date', how='right')
all = pd.merge(all, summary, on='date', how='right')

#%%
# User compliance
import pandas as pd
import matplotlib.pyplot as plt

def plot_hist_days_per_user(df, fontsize=12):
    # Group by 'User Id' and count
    user_counts = df['user id'].value_counts().sort_index()

    # Drop unwanted user IDs
    user_counts = user_counts.drop([4, 7])

    # Define the baseline (2 days for all users)
    baseline = pd.Series(2, index=user_counts.index)

    # Compute the amount above baseline
    extra = user_counts - 2

    # Prepare figure
    plt.figure(figsize=(8, 5))

    # Use categorical positions for bars (so no gaps for missing IDs)
    x_positions = range(len(user_counts))

    # Plot the baseline and the extra stacked on top
    plt.bar(x_positions, baseline, color='darkblue', edgecolor='black', label='Sessions with BFH Team')
    plt.bar(x_positions, extra, bottom=baseline, color='skyblue', edgecolor='black', label='Individual Practice')

    # Maximum per day for each user
    max_per_day = [6, 6, 6, 6, 6, 6, 5, 7, 5, 5]  # Users 1,3,5,6,8,9,10,12,13,14
    
    for idx, y in enumerate(max_per_day):
        label = 'Study Days' if idx == 0 else None
        plt.hlines(y, xmin=idx - 0.4, xmax=idx + 0.4, colors='red', linestyles='dashed', linewidth=2.5, label=label)

    # Set user IDs as tick labels
    plt.xticks(x_positions, user_counts.index, rotation=0, fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    # Title and axis labels
    plt.title('Active Robot Usage per User', fontsize=fontsize+2, fontweight='bold')
    plt.xlabel('User ID', fontsize=fontsize)
    plt.ylabel('Number of Days', fontsize=fontsize)

    # Grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), framealpha=0, ncol=3, fontsize=fontsize-2)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig("figures/user compliance.pdf", format="pdf")
    plt.savefig("figures/user compliance.svg", format="svg")
    plt.show()

plot_hist_days_per_user(all)

#%%
# Usage Time
import pandas as pd
import matplotlib.pyplot as plt

def plot_total_log_time_per_user_2(df, lang='en', fontsize=16):

    # Days each user participated
    days_per_user = { # study days
        1: 6,
        3: 6,
        5: 6,
        6: 6,
        8: 6,
        9: 6,
        10: 5,
        12: 7,
        13: 5,
        14: 5,
    }
    title = 'Average Daily Usage Time per User'

    # days_per_user = { # active days
    #     1: 3,
    #     3: 3,
    #     5: 5,
    #     6: 5,
    #     8: 3,
    #     9: 6,
    #     10: 5,
    #     12: 5,
    #     13: 3,
    #     14: 3,
    # }
    # title = 'Average Daily Usage Time per User on Active Days'

    # Language dictionaries
    labels = {
        'en': {
            'active': 'Active',
            'inactive': 'Inactive',
            'ylabel': 'Average Daily Usage Time (hours)',
            'xlabel': 'User ID',
            'title': title
        },
        'de': {
            'active': 'Aktive Nutzungszeit',
            'inactive': 'Inaktive Nutzungszeit',
            'ylabel': 'Protokollzeit pro Tag (Stunden)',
            'xlabel': 'Benutzer-ID',
            'title': 'Durchschnittliche tägliche Protokollzeit'
        },
        'fr': {
            'active': 'Temps actif',
            'inactive': 'Temps inactif',
            'ylabel': "Utilisation par jour (heures)",
            'xlabel': 'ID participant',
            'title': "Temps d'utilisation moyen journalier"
        }
    }

    if lang not in labels:
        lang = 'en'  # Default to English if language not recognized

    # Filter out specific users
    df_filtered = df[~df['user id'].isin([4, 7])].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])

    # Define relevant time columns to include
    relevant_modes = [
        'time in manual mode',
        'time in tasks mode',
        'time in fast navigation mode'
    ]

    # Sort user IDs
    sorted_user_ids = sorted(df_filtered['user id'].unique())

    user_data = []
    for user_id in sorted_user_ids:
        user_rows = df_filtered[df_filtered['user id'] == user_id]
        total_log_time = pd.to_numeric(user_rows['total log time'], errors='coerce').sum()
        active_mode_time = (
            user_rows[relevant_modes]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .sum()
            .sum()
        )

        days = days_per_user.get(user_id, None)
        if days:
            avg_total_log_time = total_log_time / days / 3600  # Convert to hours
            avg_active_time = active_mode_time / days / 3600

            user_data.append({
                'User Id': user_id,
                'Avg Total Log Time (hrs/day)': avg_total_log_time,
                'Avg Active Mode Time (hrs/day)': avg_active_time
            })

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(user_data))

    active_heights = [user['Avg Active Mode Time (hrs/day)'] for user in user_data]
    total_heights = [user['Avg Total Log Time (hrs/day)'] for user in user_data]
    remaining_heights = [
        max(total - active, 0)
        for total, active in zip(total_heights, active_heights)
    ]

    # Draw bars
    ax.bar(x, active_heights, width=0.6, label=labels[lang]['active'], color='#4169E1')
    ax.bar(x, remaining_heights, bottom=active_heights, width=0.6, color="#3e4347", alpha=0.3, label=labels[lang]['inactive'])

    # Annotate percentages on top of active bars
    for i, (active, inactive) in enumerate(zip(active_heights, remaining_heights)):
        total = active + inactive
        if total > 0:
            pct = active / total * 100
            ax.text(
                i, active + 0.05, f"{pct:.0f}%", ha='center', va='bottom',
                fontsize=fontsize-2, fontweight='bold', color='black'
            )

    # X and Y ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(user['User Id']) for user in user_data], fontsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)

    # Labels and title
    ax.set_ylabel(labels[lang]['ylabel'], fontsize=fontsize)
    ax.set_xlabel(labels[lang]['xlabel'], fontsize=fontsize)
    ax.set_title(labels[lang]['title'], fontsize=fontsize+2,fontweight='bold')

    # Y-axis limit and grid
    ax.set_ylim(0, 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), framealpha=0, ncol=2, fontsize=fontsize-2)

    plt.tight_layout()
    plt.savefig("figures/usage time.pdf", format="pdf")
    plt.savefig("figures/usage time.svg", format="svg")
    # plt.savefig("figures/usage time active days.pdf", format="pdf")
    plt.show()

    # Summary table
    df_summary = pd.DataFrame(user_data)
    df_summary['Pct Active Time'] = (
        df_summary['Avg Active Mode Time (hrs/day)'] /
        df_summary['Avg Total Log Time (hrs/day)'] * 100
    )

    # Print averages
    print(f"average total log time: {df_summary['Avg Total Log Time (hrs/day)'].mean():.2f} hrs/day")
    print(f"average active log time: {df_summary['Avg Active Mode Time (hrs/day)'].mean():.2f} hrs/day")
    print(f"average inactive log time: {(df_summary['Avg Total Log Time (hrs/day)'].mean() - df_summary['Avg Active Mode Time (hrs/day)'].mean()):.2f} hrs/day")
    print(f"average active time per log time: {(df_summary['Avg Active Mode Time (hrs/day)'].mean() / df_summary['Avg Total Log Time (hrs/day)'].mean() * 100):.2f}%")

    return df_summary

plot_total_log_time_per_user_2(all)

#%%
# Active Usage
import pandas as pd
import matplotlib.pyplot as plt

def plot_mode_count_distribution_per_user(df, fontsize=16):
    # Exclude specific users
    df_filtered = df[~df['user id'].isin([ 4, 7])].copy()

    # Sort users by their earliest date
    sorted_user_ids = sorted(df_filtered['user id'].unique())

    # Define mode columns
    mode_cols = ['manual start', 'fast start', 'tasks start']

    # Sum total time per user
    df_summed = df_filtered.groupby('user id')[mode_cols].sum().reset_index()

    # Normalize to percentage of time per user
    df_summed['total'] = df_summed[mode_cols].sum(axis=1)
    for col in mode_cols:
        df_summed[col] = df_summed[col] / df_summed['total'] * 100

    # Melt for plotting
    df_melted = df_summed.melt(
        id_vars='user id',
        value_vars=mode_cols,
        var_name='mode',
        value_name='percentage'
    )

    # Mapping for legend
    legend_mapping = {
        'manual start': 'Manual Mode',
        'fast start': 'Fast Navigation',
        'tasks start': 'Tasks Mode'
    }
    df_melted['mode'] = df_melted['mode'].replace(legend_mapping)

    # Desired order of columns
    mode_cols_pretty = ['Manual Mode', 'Fast Navigation', 'Tasks Mode']

    # Pivot for plotting
    df_pivot = df_melted.pivot(index='user id', columns='mode', values='percentage')
    df_pivot = df_pivot.loc[df_pivot.index.intersection(sorted_user_ids)]
    df_pivot = df_pivot[mode_cols_pretty]

    # Figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors
    colors = ["#ff7f0e", "#2aa02a", "#619BD4"]  # orange, green, blue

    # Stacked bar chart
    df_pivot.plot(kind='bar', stacked=True, color=colors, ax=ax)

    # Annotate each stack
    for i, user_id in enumerate(df_pivot.index):
        bottom = 0
        for mode in mode_cols_pretty:
            value = df_pivot.loc[user_id, mode]
            if value > 0:
                ax.text(
                    i,
                    bottom + value / 2,  # middle of the stack
                    f"{value:.0f}%",
                    ha='center',
                    va='center',
                    fontsize=fontsize-2,
                    fontweight='bold',
                    color='black'
                )
            bottom += value

    # Labels and formatting
    ax.set_ylabel("Percentage of Usage (%)", fontsize=fontsize)
    ax.set_xlabel('User ID', fontsize=fontsize)
    ax.set_xticklabels(df_pivot.index, rotation=0, fontsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.set_title("Distribution of Active Usage", fontsize=fontsize+2, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              framealpha=0, ncol=3, fontsize=fontsize-2)
    ax.set_ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=1)
    plt.tight_layout()
    plt.savefig("figures/active usage.pdf", format="pdf")
    plt.savefig("figures/active usage.svg", format="svg")
    plt.show()

    # Summary statistics
    for mode in mode_cols_pretty:
        mode_data = df_melted[df_melted['mode'] == mode]['percentage']
        print(f"{mode}: mean {mode_data.mean():.2f}%, min {mode_data.min():.2f}%, max {mode_data.max():.2f}%")

    return df_melted

plot_mode_count_distribution_per_user(all)

#%%
# Active Mode Completion
import matplotlib.pyplot as plt

def plot_mode_count_distribution(df, fontsize=12):
    # New order
    task_groups = ['manual', 'fast', 'tasks']

    # Mapping for display names
    display_names = {
        'tasks': 'Tasks Mode',
        'manual': 'Manual Mode',
        'fast': 'Fast Navigation'
    }

    starts = []
    stops = []
    returns = []

    for group in task_groups:
        start_val = df[f'{group} start'].sum()
        stop_val = df[f'{group} stop'].sum()
        return_val = df[f'{group} return'].sum()

        starts.append(start_val)
        stops.append(stop_val)
        returns.append(return_val)

    # Compute layer values
    finished_vals = [start - stop for start, stop in zip(starts, stops)]
    return_vals = returns
    stop_vals = [stop - ret for stop, ret in zip(stops, returns)]

    # Total height per bar
    totals = [f + r + s for f, r, s in zip(finished_vals, return_vals, stop_vals)]

    x = range(len(task_groups))
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars
    ax.bar(x, finished_vals, label='Finished', color='skyblue')
    ax.bar(x, return_vals, bottom=finished_vals, label='Stopped after entering', color='orange')
    bottom_vals = [f + r for f, r in zip(finished_vals, return_vals)]
    ax.bar(x, stop_vals, bottom=bottom_vals, label='Stopped', color='salmon')

    # Annotate percentages
    for i in x:
        layers = [finished_vals[i], return_vals[i], stop_vals[i]]
        bottom = 0
        for val in layers:
            if val > 0:
                perc = val / totals[i] * 100
                ax.text(i, bottom + val / 2, f'{perc:.0f}%', ha='center', va='center', color='black', fontsize=fontsize-2, fontweight='bold')
            bottom += val

    ax.set_title('Active Mode Completion Distribution', fontsize=fontsize + 2, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[group] for group in task_groups], fontsize=fontsize) # no xlabel
    ax.set_ylabel('Occurrences', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=3, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, -0.15))
    plt.savefig("figures/active mode completion.pdf", format="pdf")
    plt.savefig("figures/active mode completion.svg", format="svg")
    plt.show()

    return pd.DataFrame({
        'mode': ['manual', 'fast', 'tasks'],
        'finished': finished_vals,
        'returned': return_vals,
        'stopped': stop_vals,
        'total': starts
    })

plot_mode_count_distribution(all)

#%%
# Completion Analysis 1
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_total_tasks(df, prefix='total:', fontsize=14):
    """
    Plot a single 'total:' task distribution with finished/stopped and percentages.

    Parameters:
    - df: pandas DataFrame with task data
    - prefix: string, default 'total:'
    - fontsize: int, font size for all labels
    """
    
    # Filter columns that start with prefix
    prefix_cols = ['total: grab', 'total: grab finished', 'total: grab stopped', 'total: push', 'total: push finished', 'total: push stopped', 'total: next', 'total: next finished', 'total: next stopped']

    # Identify all sub-actions
    sub_actions = sorted(set(
        col[len(prefix):].strip().replace(' finished', '').replace(' stopped', '')
        for col in prefix_cols
    ))

    # Prepare data
    finished_vals = []
    stopped_vals = []

    for action in sub_actions:
        base_col = f"{prefix} {action}"
        finished_col = f"{prefix} {action} finished"
        stopped_col = f"{prefix} {action} stopped"

        if finished_col in df.columns and stopped_col in df.columns:
            finished = df[finished_col].sum()
            stopped = df[stopped_col].sum()
        elif base_col in df.columns:
            finished = 0
            stopped = df[base_col].sum()
        else:
            continue

        finished_vals.append(finished)
        stopped_vals.append(stopped)

    # Plot
    x = range(len(sub_actions))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, finished_vals, label='finished', color='skyblue')
    ax.bar(x, stopped_vals, bottom=finished_vals, label='stopped', color='salmon')

    # Annotate percentages for both finished and stopped
    for i, (f, s) in enumerate(zip(finished_vals, stopped_vals)):
        total = f + s
        if total > 0:
            pct_finished = f / total * 100
            pct_stopped = s / total * 100

            # Position labels in the middle of their respective bars
            if f > 0:
                ax.text(i, f / 2, f'{pct_finished:.0f}%', ha='center', va='center',
                        fontsize=fontsize-2, color='black', fontweight='bold')
            if s > 0:
                ax.text(i, f + s / 2, f'{pct_stopped:.0f}%', ha='center', va='center',
                        fontsize=fontsize-2, color='black', fontweight='bold')

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(sub_actions, rotation=0, fontsize=fontsize)
    ax.set_ylabel('Occurrences', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.set_title('Tasks Distribution', fontsize=fontsize+2, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=2, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig("figures/task completion 1.pdf", format="pdf")
    plt.savefig("figures/task completion 1.svg", format="svg")
    plt.show()

    # Print summary
    print(f'finished: {finished_vals}')
    print(f'stopped: {stopped_vals}')

plot_total_tasks(all)

#%%
# Completion Analysis 2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_subtasks(df, prefixes=None, fontsize=14, ylim=180):
    """
    Plot multiple subtasks in a 2x2 grid with percentages.

    Parameters:
    - df: pandas DataFrame with task data
    - prefixes: list of 4 prefixes to plot (default ['grab:', 'grab end:', 'push:', 'next:'])
    - fontsize: int, font size for all labels
    - ylim: int, max y-axis limit for all subplots
    """
    if prefixes is None:
        prefixes = ['grab:', 'grab end:', 'push:', 'next:']

    def plot_tasks_mode_ax(df, prefix, ax, title):
        # Filter columns
        prefix_cols = [col for col in df.columns if col.startswith(prefix)]

        # Identify all sub-actions
        sub_actions = sorted(set(
            col[len(prefix):].strip().replace(' finished', '').replace(' stopped', '')
            for col in prefix_cols
        ))

        # Prepare data
        finished_vals = []
        stopped_vals = []

        for action in sub_actions:
            base_col = f"{prefix} {action}"
            finished_col = f"{prefix} {action} finished"
            stopped_col = f"{prefix} {action} stopped"

            if finished_col in df.columns and stopped_col in df.columns:
                finished = df[finished_col].sum()
                stopped = df[stopped_col].sum()
            elif base_col in df.columns:
                finished = 0
                stopped = df[base_col].sum()
            else:
                continue

            finished_vals.append(finished)
            stopped_vals.append(stopped)

        # Plot stacked bars
        x = range(len(sub_actions))
        ax.bar(x, finished_vals, label='finished', color='skyblue')
        ax.bar(x, stopped_vals, bottom=finished_vals, label='stopped', color='salmon')

        # Annotate percentages for both finished and stopped
        for i, (f, s) in enumerate(zip(finished_vals, stopped_vals)):
            total = f + s
            if total > 0:
                pct_finished = f / total * 100
                pct_stopped = s / total * 100

                # Position labels in the middle of their respective bars
                min_height = 10  

                # Finished label
                if f > 0:
                    y_finished = f / 2
                    if f < min_height:
                        y_finished = f + 2  # place slightly above the finished bar
                        
                    if f < 15 and pct_finished != 100:
                        ha='right'
                    else:
                        ha='center'
                    ax.text(i, y_finished, f'{pct_finished:.0f}%', ha=ha, va='center',
                            fontsize=fontsize-2, color='black', fontweight='bold')
                # Stopped label
                if s > 0:
                    y_stopped = f + s / 2
                    if s < min_height:
                        y_stopped = f + s + 2  # place slightly above the stopped bar
                    if s < 15 and pct_stopped != 100:
                        ha='left'
                    else:
                        ha='center'
                    ax.text(i, y_stopped, f'{pct_stopped:.0f}%', ha=ha, va='center',
                            fontsize=fontsize-2, color='black', fontweight='bold')

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(sub_actions, rotation=0, fontsize=fontsize) # no xlabel
        ax.set_ylabel('Occurrences', fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize-2)
        ax.set_title(title, fontsize=fontsize+2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, ylim)

        print(ax)
        print(finished_vals)
        print(stopped_vals)
        print()

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    titles = ['Grab Actions', 'Next (after Grab) Actions', 'Push Actions', 'Next Actions']

    for i, prefix in enumerate(prefixes):
        ax = axes[i // 2, i % 2]
        plot_tasks_mode_ax(df, prefix, ax, titles[i])

    # Single horizontal legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig("figures/task completion 2.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("figures/task completion 2.svg", format="svg", bbox_inches="tight")
    plt.show()

plot_subtasks(all)

#%%
# Dropout Analysis & Detection Analysis

print(f'''
detection: grab unable: {all['detection: grab unable'].sum()} = 23
detection: grab bad: {all['detection: grab bad'].sum()} = 12
detection: grab stopped: {all['detection: grab stopped'].sum()} = 3
detection: grab stopped after: {all['detection: grab stopped after'].sum()} = 9
detection: grab good: {all['detection: grab good'].sum()} = 242
detection: grab retries: {all['detection: grab retries'].sum()} = 26
detection: grab: {all['detection: grab'].sum()} = 280
detection: grab multiple options: {all['detection: grab multiple options'].sum()} = 46
detection: grab not first choice: {all['detection: grab not first choice'].sum()} = 9
detection: gripper changed: {all['detection: gripper changed'].sum()} = 49


detection: push unable: {all['detection: push unable'].sum()} = 6
detection: push bad: {all['detection: push bad'].sum()} = 6
detection: push stopped: {all['detection: push stopped'].sum()} = 4
detection: push stopped after: {all['detection: push stopped after'].sum()} = 2
detection: push good: {all['detection: push good'].sum()} = 75
detection: push retries: {all['detection: push retries'].sum()} = 10
detection: push: {all['detection: push'].sum()} = 91
detection: push multiple options: {all['detection: push multiple options'].sum()} = 50
detection: push not first choice: {all['detection: push not first choice'].sum()} = 20

grab control start: {all['control: grab start'].sum()} = 242
grab control 1 good: {all['control: grab good'].sum()} = 125
grab control 1 good first: {all['control: grab good first'].sum()} = 96
grab control 1 no: {all['control: grab bad'].sum()} = 101
grab control 1 no first: {all['control: grab bad first'].sum()} = 78
grab control 1 number of manual manip: {all['control: grab altered'].sum()} = 362
grab control 1 stop: {all['control: grab stopped'].sum()} = 10
grab control 1 stop manual: {all['control: grab stopped manual'].sum()} = 11
grab control 1 closer: {all['control: grab closer'].sum()} = 163
grab control 1 stopped first: {all['control: grab stopped first'].sum()} = 7


grab control 2 good: {all['control: grab end good'].sum()} = 121
grab control 2 no: {all['control: grab end bad'].sum()} = 90
grab control 2 number of manual manip: {all['control: grab end altered'].sum()} = 227
grab control 2 number of manual grip: {all['control: grab end gripper'].sum()} = 84
grab control 2 stop: {all['control: grab end stopped'].sum()} = 4
grab control 2 stop manual: {all['control: grab end stopped manual'].sum()} = 8

grab control finished: {all['control: grab finished'].sum()} = 203

push control start: {all['control: push start'].sum()} = 75
push control no: {all['control: push bad'].sum()} = 104
push control stopped: {all['control: push stopped'].sum()} = 18
push control stopped first: {all['control: push stopped first'].sum()} = 8
push control good: {all['control: push good'].sum()} = 57
push control good first: {all['control: push good first'].sum()} = 30
push control finished: {all['control: push finished'].sum()} = 57
''')

#%%
# Fast Navigation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_gripper_orientation_total(df, fontsize=12):
    """
    Plot gripper orientations with all 'fast' and 'tasks' values added together in a single stacked bar chart.
    The 'back' orientation is displayed fully in salmon.
    """
    prefixes = ['gripper', 'tasks gripper']
    orientation_cols = ['up', 'down', 'center', 'left', 'right', 'back']
    orientations = orientation_cols  # For labeling
    x = np.arange(len(orientation_cols))

    base_vals = []
    stopped_vals = []
    bar_types = []

    # Filter out rows where all orientation counts are 0
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered[[f'{p}: {o}' for p in prefixes for o in orientation_cols]].sum(axis=1) > 0]

    # Compute finished and stopped counts
    for orientation, base_col in zip(orientations, orientation_cols):
        stopped_total = 0
        finished_total = 0

        for prefix in prefixes:
            col = f'{prefix}: {orientation}'
            stopped_col = f'{col} stopped'

            total = df_filtered[col].sum() if col in df_filtered.columns else 0
            stopped = df_filtered[stopped_col].sum() if stopped_col in df_filtered.columns else 0
            finished = max(total - stopped, 0)

            finished_total += finished
            stopped_total += stopped

        # Make 'back' fully salmon
        if orientation == 'back':
            stopped_total = finished_total + stopped_total
            finished_total = 0
            bar_types.append('salmon_only')
        else:
            if finished_total > 0 and stopped_total > 0:
                bar_types.append('stacked')
            else:
                bar_types.append('salmon_only')

        base_vals.append(finished_total)
        stopped_vals.append(stopped_total)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, base_vals, label='Finished', color='skyblue')
    ax.bar(x, stopped_vals, bottom=base_vals, label='Stopped', color='salmon')

    # Percent annotations
    for i, (f, s) in enumerate(zip(base_vals, stopped_vals)):
        total = f + s
        if total > 0:
            if f > 0:
                ax.text(i, f/2, f'{f/total*100:.0f}%', ha='center', va='center',
                        fontsize=fontsize-2, fontweight='bold')
            if s > 0:
                ax.text(i, f + s/2, f'{s/total*100:.0f}%', ha='center', va='center',
                        fontsize=fontsize-2, fontweight='bold')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(orientations, fontsize=fontsize)
    ax.set_ylabel('Occurrences', fontsize=fontsize)
    ax.set_title('Gripper Orientations in Fast Navigation', fontsize=fontsize+2, fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    ax.legend(loc='lower center', ncol=2, fontsize=fontsize-2, framealpha=0, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig("figures/gripper.pdf", format="pdf")
    plt.savefig("figures/gripper.svg", format="svg")
    plt.show()

    print('Finished counts (all except back):', base_vals)
    print('Stopped counts (including back):', stopped_vals)

plot_gripper_orientation_total(all)

#%%
# Stop Usage 1
import matplotlib.pyplot as plt

print(f'''
overall tasks {all['overall tasks'].sum()} = 621
tasks start {all['tasks start'].sum()} = 621
overall tasks stopped {all['overall tasks stopped'].sum()} = 356 # no back
tasks stop {all['tasks stop'].sum()} = 376

tasks: GUI Stop {all['tasks: GUI Stop'].sum()} = 290
tasks: GUI Stop Moving {all['tasks: GUI Stop Moving'].sum()} = 27
tasks: Head Stop {all['tasks: Head Stop'].sum()} = 8
tasks: Head Stop Moving {all['tasks: Head Stop Moving'].sum()} = 8
tasks: Manual Stop {all['tasks: Manual Stop'].sum()} = 58
tasks: Startup {all['tasks: Startup'].sum()} = 1

gripper: total rows {all['gripper: total rows'].sum()} = 124
fast start {all['fast start'].sum()} = 124
fast stop {all['fast stop'].sum()} = 65

fast: GUI Stop {all['fast: GUI Stop'].sum()} = 2
fast: GUI Stop Moving {all['fast: GUI Stop Moving'].sum()} = 2
fast: Head Stop {all['fast: Head Stop'].sum()} = 2
fast: Head Stop Moving {all['fast: Head Stop Moving'].sum()} = 2
fast: Manual Stop {all['fast: Manual Stop'].sum()} = 21
fast: Startup {all['fast: Startup'].sum()} = 1


tasks start {all['tasks start'].sum()} = 621
tasks stop {all['tasks stop'].sum()} = 376
fast start {all['fast start'].sum()} = 124
fast stop {all['fast stop'].sum()} = 65
home: total {all['home: total'].sum()} = 758
home: stopped {all['home: stopped'].sum()} = 110
table: total {all['table: total'].sum()} = 109
table: stopped {all['table: stopped'].sum()} = 13
drive: total {all['drive: total'].sum()} = 357
drive: stopped {all['drive: stopped'].sum()} = 20
''')


def plot_modes(df, fontsize=12):
    """
    Creates a stacked bar chart for modes:
    Tasks, Fast Navigation, Home, Table, Drive.
    
    fontsize: base font size
        - title uses fontsize + 2
        - axes labels, ticks, legend use fontsize - 2
        - bar text uses fontsize
    """

    # Mapping dataframe column names to mode totals and stops
    modes = {
        "Tasks": {
            "total": df["tasks start"].sum(),
            "stop": df["tasks stop"].sum()
        },
        "Fast Navigation": {
            "total": df["fast start"].sum(),
            "stop": df["fast stop"].sum()
        },
        "Home": {
            "total": df["home: total"].sum(),
            "stop": df["home: stopped"].sum()
        },
        "Table": {
            "total": df["table: total"].sum(),
            "stop": df["table: stopped"].sum()
        },
        "Drive": {
            "total": df["drive: total"].sum(),
            "stop": df["drive: stopped"].sum()
        },
    }

    labels = list(modes.keys())
    totals = [modes[m]["total"] for m in labels]
    stops  = [modes[m]["stop"] for m in labels]
    remaining = [tot - st for tot, st in zip(totals, stops)]

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    
    # Bottom stack: remaining
    bars1 = plt.bar(x, remaining, color='skyblue', label="Finished")
    # Top stack: stopped
    bars2 = plt.bar(x, stops, bottom=remaining, color='salmon', label="Stopped")

    # Add percentages inside bars
    for i, (rem, st, tot) in enumerate(zip(remaining, stops, totals)):
        # Remaining percentage
        if rem > 0:
            plt.text(
                i, rem/2, 
                f"{rem / tot * 100:.0f}%", 
                ha="center", va="center",
                fontsize=fontsize-2, color='black', fontweight='bold'
            )
        # Stopped percentage
        if st > 0:
            # If very small, move text outside
            if st < 50:
                y_pos = rem + st + 15 
            else:
                y_pos = rem + st/2
            plt.text(
                i, y_pos,
                f"{st / tot * 100:.0f}%",
                ha="center", va="center",
                fontsize=fontsize-2, color='black', fontweight='bold'
            )

    plt.xticks(x, labels, fontsize=fontsize) # no xlabel
    plt.yticks(fontsize=fontsize-2)
    plt.ylabel("Occurrences", fontsize=fontsize)
    plt.title("Stop Usage per Mode", fontsize=fontsize + 2, fontweight='bold')

    plt.legend(
        loc='lower center', ncol=2, fontsize=fontsize-2,
        framealpha=0, bbox_to_anchor=(0.5, -0.15)
    )

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/safety 1.pdf", format="pdf")
    plt.savefig("figures/safety 1.svg", format="svg")
    plt.show()


plot_modes(all)

#%% percent
# Stop Usage 2

def plot_stop_breakdown_stacked(df, fontsize=12):
    """
    Stacked bar plot for tasks stop and fast stop.
    GUI Stop Moving is a sub-stack of GUI Stop.
    Head Stop Moving is a sub-stack of Head Stop.
    """

    # ----- Categories including sub-stacks -----
    categories = [
        "Head Stop", "Head Stop Moving",
        "GUI Stop", "GUI Stop Moving",
        "Manual Stop", "Other"
    ]

    # ----- TASKS -----
    gui_static_tasks  = df["tasks: GUI Stop"].sum() - df["tasks: GUI Stop Moving"].sum()
    gui_moving_tasks  = df["tasks: GUI Stop Moving"].sum()

    head_static_tasks = df["tasks: Head Stop"].sum() - df["tasks: Head Stop Moving"].sum()
    head_moving_tasks = df["tasks: Head Stop Moving"].sum()

    manual_tasks      = df["tasks: Manual Stop"].sum()

    total_tasks = df["tasks stop"].sum()

    other_tasks = total_tasks - (
        gui_static_tasks + gui_moving_tasks +
        head_static_tasks + head_moving_tasks +
        manual_tasks
    )

    tasks_vals = [
        head_static_tasks, head_moving_tasks,
        gui_static_tasks, gui_moving_tasks,
        manual_tasks, other_tasks
    ]

    # ----- FAST -----
    gui_static_fast  = df["fast: GUI Stop"].sum() - df["fast: GUI Stop Moving"].sum()
    gui_moving_fast  = df["fast: GUI Stop Moving"].sum()

    head_static_fast = df["fast: Head Stop"].sum() - df["fast: Head Stop Moving"].sum()
    head_moving_fast = df["fast: Head Stop Moving"].sum()

    manual_fast      = df["fast: Manual Stop"].sum()

    total_fast = df["fast stop"].sum()

    other_fast = total_fast - (
        gui_static_fast + gui_moving_fast +
        head_static_fast + head_moving_fast +
        manual_fast
    )

    fast_vals = [
        head_static_fast, head_moving_fast,
        gui_static_fast, gui_moving_fast,
        manual_fast, other_fast
    ]

    # ----- Percentages -----
    tasks_pct = [v / total_tasks * 100 for v in tasks_vals]
    fast_pct  = [v / total_fast  * 100 for v in fast_vals]

    # ----- PRINT PERCENTAGES -----
    print("\n=== Percentage Breakdown ===")

    print("\nTasks:")
    for c, p in zip(categories, tasks_pct):
        print(f"  {c:16s}: {p:5.2f}%")

    print("\nFast Navigation:")
    for c, p in zip(categories, fast_pct):
        print(f"  {c:16s}: {p:5.2f}%")
    print()

    # ----- PLOT -----
    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = {"Tasks": 0, "Fast Navigation": 0}

    # ----- Reorder colors to match new categories -----
    colors = [
        "#55a868", "#9ed8a4",   # Head: static + moving
        "#4c72b0", "#8db0e3",   # GUI: static + moving
        "#8172b3",              # Manual
        "#ccb974",              # Other
    ]

    # Draw stacks
    for i, cat in enumerate(categories):
    
        # ----- Tasks column -----
        ax.bar("Tasks", tasks_pct[i], bottom=bottoms["Tasks"], color=colors[i])

        # Only show label if > 0
        if tasks_pct[i] > 1:
            ax.text(
                "Tasks",
                bottoms["Tasks"] + tasks_pct[i] / 2,
                f"{tasks_pct[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=fontsize-2,
                color='black',
                fontweight='bold'
            )

        elif tasks_pct[i] > 0 and tasks_pct[i] < 1:
            ax.text(
                "Tasks",
                bottoms["Tasks"] + tasks_pct[i] / 2,
                f"{tasks_pct[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize-2,
                color='black',
                fontweight='bold'
            )

        bottoms["Tasks"] += tasks_pct[i]

        # ----- Fast column -----
        ax.bar("Fast Navigation", fast_pct[i], bottom=bottoms["Fast Navigation"], color=colors[i], label=cat)

        # Only show label if > 0
        if fast_pct[i] > 1:
            ax.text(
                "Fast Navigation",
                bottoms["Fast Navigation"] + fast_pct[i] / 2,
                f"{fast_pct[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=fontsize-2,
                color='black',
                fontweight='bold'
            )

        elif fast_pct[i] > 0 and fast_pct[i] < 1:
            ax.text(
                "Fast Navigation",
                bottoms["Fast Navigation"] + fast_pct[i] / 2,
                f"{fast_pct[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=fontsize-2,
                color='black',
                fontweight='bold'
            )

        bottoms["Fast Navigation"] += fast_pct[i]

    ax.set_ylabel("Percentage (%)", fontsize=fontsize)
    ax.set_title("Stop Breakdown", fontsize=fontsize+2, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.savefig("figures/safety 2.pdf", format="pdf",bbox_inches="tight")
    plt.savefig("figures/safety 2.svg", format="svg", bbox_inches="tight")
    plt.show()

plot_stop_breakdown_stacked(all)

#%%
# Restarts
print(f'''
settings restart: restarts {all['settings restart: restarts'].sum()} = 146
settings shutdown: restarts {all['settings shutdown: restarts'].sum()} = 6
settings restart: shutdowns {all['settings restart: shutdowns'].sum()} = 8
settings shutdown: shutdowns {all['settings shutdown: shutdowns'].sum()} = 26

total: restarts: {all['total: restarts'].sum()} = 192
total: shutdowns: {all['total: shutdowns'].sum()} = 67

total: restarts + shutdowns: {all['total: restarts + shutdowns'].sum()} = 259
''')

def plot_system_events(df, fontsize=12):
    modes = ["settings restart", "settings shutdown"]

    under = {m: df.get(f"{m}: restarts", pd.Series([0])).sum() for m in modes}
    over  = {m: df.get(f"{m}: shutdowns", pd.Series([0])).sum() for m in modes}

    under["other"] = df["total: restarts"].sum() - sum(under.values())
    over["other"]  = df["total: shutdowns"].sum() - sum(over.values())

    restart_vals  = np.array([under[m] for m in under])
    shutdown_vals = np.array([over[m] for m in over])

    x = np.arange(len(under))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_restart  = ax.bar(x - width/2, restart_vals,  width,
                           label="< 15 minutes", color="tab:blue")
    bars_shutdown = ax.bar(x + width/2, shutdown_vals, width,
                           label="≥ 15 minutes", color="tab:orange")

    # Percentages
    total_restarts  = restart_vals.sum()
    total_shutdowns = shutdown_vals.sum()

    restart_pct  = restart_vals  / total_restarts  * 100
    shutdown_pct = shutdown_vals / total_shutdowns * 100

    # Percentage labels above bars
    for i, bar in enumerate(bars_restart):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{restart_pct[i]:.0f}%", ha="center", va="bottom",
                fontsize=fontsize-2, fontweight="bold", color="tab:blue")

    for i, bar in enumerate(bars_shutdown):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{shutdown_pct[i]:.0f}%", ha="center", va="bottom",
                fontsize=fontsize-2, fontweight="bold", color="tab:orange")

    # Prettier x-labels
    pretty_labels = [
        "Restart Button\n(settings)",
        "Shutdown Button\n(settings)",
        "Power Cut"
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, fontsize=fontsize)

    # Titles
    ax.set_ylabel("Occurrences", fontsize=fontsize)
    ax.set_title("System Restart & Shutdown Events", fontsize=fontsize+4, weight="bold")

    # Y-axis tick font
    ax.tick_params(axis="y", labelsize=fontsize-2)

    # Legend
    legend = ax.legend(
        title="Duration of Downtime:",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        framealpha=0,
        ncol=2,
        fontsize=fontsize-2
    )
    legend.get_title().set_fontsize(fontsize-2)
    legend.get_title().set_fontweight("bold")

    # Light grid
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("figures/reliability.pdf", format="pdf")
    plt.savefig("figures/reliability.svg", format="svg")
    plt.show()

plot_system_events(all, 14)

#xyz
print(f'total: restarts without dropouts: {all.loc[~all['user id'].isin([4, 7]), 'total: restarts'].sum()}')
print(f'number of logs without dropouts and empty: {all.loc[~all['user id'].isin([4, 7])].shape[0]}')
print(f'manual: restarts: {all["manual: restarts"].sum()}')
print(f'home: restarts: {all["home: restarts"].sum()}')
print(f'homepage: restarts: {all["homepage: restarts"].sum()}')

#%%
# Supplementary Figure 2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_manual_in_tasks(counter1: Counter, counter2: Counter, fontsize=12):
    """
    Two stacked bar charts with shared y-axis, percentages based on total counts,
    and one legend at the bottom.
    """
    movement_order = ["down", "up", "forward", "backward", "left", "right"]
    gripper_rotation_order = [
        "Gripper Close", "Gripper Open", "rollNegative", "rollPositive",  
        "yawNegative", "yawPositive", "pitchNegative", "pitchPositive"
    ]

    width = 0.38

    # Compute max y-value
    all_counts = []
    for order in [movement_order, gripper_rotation_order]:
        values1 = np.array([counter1.get(label, 0) for label in order])
        values2 = np.array([counter2.get(label, 0) for label in order])
        all_counts.extend(values1)
        all_counts.extend(values2)
    ymax = max(all_counts) * 1.1

    # Totals for percentages across both plots
    total1 = sum(counter1.get(cmd,0) for cmd in movement_order + gripper_rotation_order)
    total2 = sum(counter2.get(cmd,0) for cmd in movement_order + gripper_rotation_order)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # --- Top plot: movement commands ---
    values1 = np.array([counter1.get(label, 0) for label in movement_order])
    values2 = np.array([counter2.get(label, 0) for label in movement_order])
    x = np.arange(len(movement_order))

    bars1 = axes[0].bar(x - width/2, values1, width, color="royalblue")
    bars2 = axes[0].bar(x + width/2, values2, width, color="tomato")

    pct1 = values1 / total1 * 100
    pct2 = values2 / total2 * 100

    for i, bar in enumerate(bars1):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{pct1[i]:.0f}%", ha="center", va="bottom",
                     fontsize=fontsize-2, fontweight="bold", color="royalblue")
    for i, bar in enumerate(bars2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{pct2[i]:.0f}%", ha="center", va="bottom",
                     fontsize=fontsize-2, fontweight="bold", color="tomato")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(movement_order, fontsize=fontsize)
    axes[0].set_ylabel("Occurrences", fontsize=fontsize)
    axes[0].tick_params(axis="y", labelsize=fontsize-2)
    axes[0].set_title("Direction Commands", fontsize=fontsize+2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].set_ylim(0, ymax)

    # --- Bottom plot: gripper/rotation commands ---
    values1 = np.array([counter1.get(label, 0) for label in gripper_rotation_order])
    values2 = np.array([counter2.get(label, 0) for label in gripper_rotation_order])
    x = np.arange(len(gripper_rotation_order))

    bars1 = axes[1].bar(x - width/2, values1, width, color="royalblue")
    bars2 = axes[1].bar(x + width/2, values2, width, color="tomato")

    pct1 = values1 / total1 * 100
    pct2 = values2 / total2 * 100

    for i, bar in enumerate(bars1):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{pct1[i]:.0f}%", ha="center", va="bottom",
                     fontsize=fontsize-2, fontweight="bold", color="royalblue")
    for i, bar in enumerate(bars2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{pct2[i]:.0f}%", ha="center", va="bottom",
                     fontsize=fontsize-2, fontweight="bold", color="tomato")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gripper_rotation_order, fontsize=fontsize)
    axes[1].set_ylabel("Occurrences", fontsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize-2)
    axes[1].set_title("Gripper & Rotation Commands", fontsize=fontsize+2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].set_ylim(0, ymax)

    # --- Shared legend at the bottom ---
    fig.legend(['Ready-to-Grip', 'Grip Evaluation'], 
               loc='lower center', ncol=2, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, 0.05))

    plt.savefig("figures/manual task.pdf", format="pdf")
    plt.savefig("figures/manual task.svg", format="svg")
    plt.show()

# Call the function
plot_manual_in_tasks(man1, man2, fontsize=14)

#%%
# Supplementary Figure 3
import matplotlib.pyplot as plt
import numpy as np

def plot_manual_analysis(df, fontsize=12):
    """
    Same two-panel plot, but with:
    - SUM of each column in df
    - ONE dataset (one bar per command)
    - Bar width matches grouped version (width=0.38)
    """

    movement_order = ["down", "up", "forward", "backward", "left", "right"]
    gripper_rotation_order = [
        "Gripper Close", "Gripper Open", "rollNegative", "rollPositive",
        "yawNegative", "yawPositive", "pitchNegative", "pitchPositive"
    ]

    # bar width identical to your grouped compare plot
    width = 0.38

    # --- Sum all columns ---
    col_sums = df.sum()

    mov_values = np.array([col_sums[f"manual: {cmd}"] for cmd in movement_order])
    grip_values = np.array([col_sums[f"manual: {cmd}"] for cmd in gripper_rotation_order])

    total = mov_values.sum() + grip_values.sum()
    ymax = max(mov_values.max(), grip_values.max()) * 1.15

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # ================= TOP PANEL =================
    x = np.arange(len(movement_order))
    bars = axes[0].bar(x, mov_values, width=width, color="forestgreen")

    pct = mov_values / total * 100
    for i, bar in enumerate(bars):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize-2,
            fontweight="bold",
            color="forestgreen"
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(movement_order, fontsize=fontsize)
    axes[0].set_ylabel("Occurrences", fontsize=fontsize)
    axes[0].tick_params(axis="y", labelsize=fontsize-2)
    axes[0].set_title("Direction Commands", fontsize=fontsize+2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].set_ylim(0, ymax)

    # ================= BOTTOM PANEL =================
    x = np.arange(len(gripper_rotation_order))
    bars = axes[1].bar(x, grip_values, width=width, color="forestgreen")

    pct = grip_values / total * 100
    for i, bar in enumerate(bars):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize-2,
            fontweight="bold",
            color="forestgreen"
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gripper_rotation_order, fontsize=fontsize)
    axes[1].set_ylabel("Occurrences", fontsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize-2)
    axes[1].set_title("Gripper & Rotation Commands", fontsize=fontsize+2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].set_ylim(0, ymax)

    fig.legend(['Manual Mode'], 
               loc='lower center', ncol=2, fontsize=fontsize-2,
               framealpha=0, bbox_to_anchor=(0.5, 0.05))

    plt.savefig("figures/manual.pdf", format="pdf")
    plt.savefig("figures/manual.svg", format="svg")
    plt.show()

    # ================= SUMMARY DATAFRAME =================
    summary_dict = {f"manual: {cmd}": col_sums[f"manual: {cmd}"] 
                    for cmd in movement_order + gripper_rotation_order}
    summary_dict['Total'] = total
    summary_df = pd.DataFrame(summary_dict, index=['Count'])
    
    return summary_df

plot_manual_analysis(all, fontsize=14)