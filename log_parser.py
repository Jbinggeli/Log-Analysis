import os
import re
from datetime import datetime
import pandas as pd

MANUAL_BUTTONS = ['Continous down', 'Continous up',
                  'Continous right', 'Continous left',
                  'Continous forward', 'Continous backward',
                  'Continous pitchNegative', 'Continous pitchPositive',
                  'Continous yawNegative', 'Continous yawPositive',
                  'Continous rollNegative', 'Continous rollPositive',
                  'Gripper Open', 'Gripper Close']

STEPWISE_BUTTONS = ['Stepwise down', 'Stepwise up',
                  'Stepwise right', 'Stepwise left',
                  'Stepwise forward', 'Stepwise backward',
                  'Stepwise pitchNegative', 'Stepwise pitchPositive',
                  'Stepwise yawNegative', 'Stepwise yawPositive',
                  'Stepwise rollNegative', 'Stepwise rollPositive']

# ================================
# Pandas Display Settings
# ================================
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('expand_frame_repr', False)


# ================================
# Timestamp Parsing
# ================================
def parse_ts(ts_str):
    """Parse timestamp from log string into datetime."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")

# ================================
# Utility: Save Mode CSVs
# ================================
def save_mode_csvs(modes_dict, folder_path, prefix):
    """Save mode DataFrames as CSV files with the given prefix."""
    for mode_name, df in modes_dict.items():
        filename = f"{prefix}_{mode_name}.csv"
        df.to_csv(os.path.join(folder_path, filename), index=False)

def get_level(line):
    # Case 1: Check for "rendering page"
    if "rendering page" in line:
        parts = line.split()
        try:
            idx = parts.index("page")
            return parts[idx + 1]  # return the word after "page"
        except (ValueError, IndexError):
            pass  # "page" not found or no word after it

    # Case 2: Check for run navigate(('prompt',), {'prompt':
    match = re.search(r"run navigate\(.*?'prompt':\s*'(.*?)'", line)
    if match:
        return match.group(1)

    return None  # If nothing is found

# ================================
# Log Parsing Function
# ================================
def get_informative_df(lines):
    """
    Parse raw log lines to extract timestamps, button presses, 
    system startups, and return-to-home events.
    Input:
        lines: from file.readlines()
    Output:
        DataFrame with:
        - idx: line number
        - timestamp: datetime
        - button: name of button pressed
        - startup: True if system start
        - back_to_homescreen: True if returned to homescreen
        - line: full line of log
    """
    timestamp_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    button_press_re = re.compile(r"button_press\s*\(\s*['\"](.*?)['\"]")
    startup_re = re.compile(r"loading settings")
    back_re = re.compile(r"Back at Level1")

    data = []
    for idx, line in enumerate(lines):
        ts_match = timestamp_re.match(line)
        if not ts_match:
            continue

        timestamp = parse_ts(ts_match.group(1))
        button_match = button_press_re.search(line)
        level = get_level(line)

        data.append({
            'idx': idx,
            'timestamp': timestamp,
            'button': button_match.group(1) if button_match else None,
            'startup': bool(startup_re.search(line)),
            'back_to_homescreen': bool(back_re.search(line)),
            'level': level,
            'line': line.strip()
        })

    return pd.DataFrame(data)

# ================================
# Basic Counters
# ================================
def startup_counter(df, date, printout=False):
    """
    Count the number of system startups.
    Input:
        df: df from get_informative_df()
    Output:
        integer
    """
    count = df['startup'].sum()

    if printout:
        print(f"Number of system startups: {count}")


    # Mode files map
    mode_files = {
        'SettingsInput': f'data/{date}/{date}_settingsMode.csv',
        'FastNavigationInput': f'data/{date}/{date}_fastMode.csv',
        'TasksInput': f'data/{date}/{date}_tasksMode.csv',
        'ManualInput': f'data/{date}/{date}_manualMode.csv',
        'HomeInput': f'data/{date}/{date}_homeMode.csv',
        'TableAccessInput': f'data/{date}/{date}_tableMode.csv',
        'DriveInput': f'data/{date}/{date}_driveMode.csv'
    }

    # Load all CSVs once
    df_modes = {}
    for btn, file in mode_files.items():
        try:
            df_modes[btn] = pd.read_csv(file)
        except Exception:
            df_modes[btn] = pd.DataFrame()

    valid_buttons = list(df_modes.keys())

    results = []

    # -------------------------------
    # Helper: Find mode from last button
    # -------------------------------
    def find_mode_from_button(btn, timestamp, search):
        """Return mode name given a button + timestamp."""
        df_mode = df_modes.get(btn, pd.DataFrame())
        if df_mode.empty:
            return "Homepage"

        if "timestamp" in df_mode.columns:
            df_mode["timestamp"] = pd.to_datetime(df_mode["timestamp"])

        row = df_mode[df_mode["timestamp"] == timestamp]
        if row.empty:
            return "Homepage"

        if btn in ["HomeInput", "TableAccessInput", "DriveInput"]:
            if row["next"].iloc[0] == search:
                return btn.replace("Input", "")
        else:
            if row["exit"].iloc[0] == search:
                return btn.replace("Input", "")

        return "Homepage"

    # ---------------------------------
    # PART 1 — Process each startup
    # ---------------------------------

    for i, row in df.iterrows():
        if not row['startup']:
            continue

        if i == 0:
            results.append({
                "timestamp": row["timestamp"],
                "turned_off": 0,
                "mode": "StartLog"
            })
            continue

        turned_off = (row["timestamp"] - df.iloc[i - 1]["timestamp"]).total_seconds()

        # Look backward for last button press
        last_btn, last_ts = None, None

        for j in range(i - 1, -1, -1):
            btn = df.iloc[j]["button"]
            start = df.iloc[j]['startup']
            if btn in valid_buttons or start: # until last Mode or Restart
                last_btn = btn
                last_ts = df.iloc[j]["timestamp"]
                break

        if last_btn:
            mode = find_mode_from_button(last_btn, last_ts, "startup")
            ts = last_ts
        else:
            mode, ts = "Homepage", row["timestamp"]

        results.append({
            "timestamp": ts,
            "turned_off": turned_off,
            "mode": mode
        })

    # ---------------------------------
    # PART 2 — Determine the mode of the last log segment
    # ---------------------------------

    last_btn = None
    last_ts = None

    # Search backward from last log entry
    for j in range(len(df) - 1, -1, -1):
        btn = df.iloc[j]["button"]
        start = df.iloc[j]['startup']
        if btn in valid_buttons or start:
            last_btn = btn
            last_ts = df.iloc[j]["timestamp"]
            break

    if last_btn:
        last_mode = find_mode_from_button(last_btn, last_ts, "End of Log")
        ts = last_ts
    else:
        last_mode, ts = "Homepage", row["timestamp"]

    # Append final mode entry
    results.append({
        "timestamp": ts,
        "turned_off": 0,
        "mode": last_mode
    })

    # ---------------------------------
    # Save and return
    # ---------------------------------

    df_out = pd.DataFrame(results)
    df_out.to_csv(f"data/{date}/{date}_Startups.csv", index=False)

    return count

# ================================
# Duration & Time Tracking
# ================================
def get_total_log_time(df, printout=False):
    """
    Calculate total log time excluding restarts.
    Input:
        df: df from get_informative_df()
    Output:
        float
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    total_time = 0.0
    last_startup_idx = 0

    for i in range(1, len(df)):
        # exclude restarts
        if df.loc[i, 'startup']:
            start_time = df.loc[last_startup_idx, 'timestamp']
            end_time = df.loc[i - 1, 'timestamp']
            total_time += (end_time - start_time).total_seconds()
            last_startup_idx = i

    # last timestamp of log file, if it was not already calculated
    if last_startup_idx < len(df) - 1:
        total_time += (df.iloc[-1]['timestamp'] - df.loc[last_startup_idx, 'timestamp']).total_seconds()

    if printout:
        print(f"{'Total Log Time':<40}: {total_time:.3f} seconds")
    return total_time

def get_total_duration(df, name='', printout=False):
    """
    Sums up duration from a DataFrame
    Input:
        df: df with column duration
    Output:
        float
    """
    if df.empty:
        return 0.0
    total = df['duration'].sum()
    if printout:
        print(f"{f'Total Duration of {name}':<40}: {total:.3f} seconds")
    return total
        

# ================================
# Mode Tracking Functions
# ================================
def track_mode(df, mode_button, mode_name, printout=False):
    """
    Track time and action spent in a mode that has sequenctial selection options like Manual, Tasks, Fast Navigation, Settings.
    Input:
        df: df from get_informative_df()
    Output:
        Dataframe with:
        - timestamp: datetime
        - buttons_pressed: integer
        - exit: #TODO
        - buttons_sequence: list with button names
        - duration: float

    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    total_time = 0.0
    mode_count = 0
    mode_data = []

    for i, row in df.iterrows():
        if row['button'] != mode_button:
            continue

        mode_count += 1
        start_time = row['timestamp']
        end_time = None
        buttons_pressed = 0
        buttons_seq = []
        levels_seq = []
        last_button = None
        turnedoff_time = 0

        if printout:
            print(f"[{start_time}] Entered {mode_name}")

        for j in range(i + 1, len(df)):
            next_row = df.iloc[j]
            next_button = next_row['button']
            next_level = next_row['level']

            # system is restarted -> end tracking
            if next_row['startup']:
                exit_mode = 'startup' 
                end_time = df.iloc[j - 1]['timestamp'] # stayed until the last log time [j-1] before system restarted in this mode
                turnedoff_time = (df.iloc[j]['timestamp'] - df.iloc[j - 1]['timestamp']).total_seconds()
                break

            # user retured to the homescreen -> end tracking
            if next_row['back_to_homescreen']:
                end_time = next_row['timestamp'] # stayed until system returned to homescreen in this mode
                if mode_button in ['TasksInput', 'FastNavigationInput']:
                    
                    if buttons_seq[-1] == 'yes':
                        exit_mode = 'finished (yes)'

                    elif buttons_seq[-1] == 'no': # fast navigation when it asks free to move? 'no' returns you to the homepage
                        exit_mode = 'stopped (no)'

                    elif buttons_seq[-1] == 'stop':

                        if levels_seq[-2:] == ['task_selection', 'stop'] or buttons_seq[-2:] == ['GUI STOP pitchNegative', 'stop']: # always accompanied by GUI stop only not in task_selection
                            exit_mode = 'GUI stop'

                        else:
                            exit_mode = 'head stop'

                    elif buttons_seq[-1] == 'done':

                        if buttons_seq[-2:] == ['next', 'done']:
                            exit_mode = 'stopped (done)'
                        
                        else: # after grab not continued to place -> 'nextSelection', 'done'
                            exit_mode = 'finished (done)'
                        
                    elif buttons_seq[-1] == 'back':
                        if levels_seq[-3:] == ['back', 'Abschliessen...', 'prompt'] or levels_seq[-3:] == ['back', 'finishing up...', 'prompt']: # from push, place on table, drink
                            exit_mode = 'finished (back)'

                        elif len(buttons_seq) > 1 and ( # from fast navigation
                            buttons_seq[-2] in MANUAL_BUTTONS 
                            or buttons_seq[-2] in STEPWISE_BUTTONS
                            or levels_seq[-2] == 'manualV1'
                        ):
                            ''''
                            moving...', 'prompt', 'manual', 'manualV1', 'back'
                            'am bewegen...', 'prompt', 'manual', 'manualV1', 'back'
                            'am bewegen...', 'prompt', 'manual', 'manualV1', 'Continous down', 'Continous up', 'back' 
                            '''
                            exit_mode = 'finished (fn)'
                        else:
                            exit_mode = 'stopped (back)'
                    
                    elif buttons_seq[-1] == 'estop':
                        exit_mode = 'manual stop'

                    else:
                        print('whut', buttons_seq)
                        exit_mode = None

                else: # settings and manual
                    exit_mode = buttons_seq[-1]
                break

            # another button is pressed -> track bottons
            if pd.notna(next_button):
                if next_button in MANUAL_BUTTONS and next_button == last_button:
                    continue              
                if printout:
                    print(f"  → {next_button}")
                buttons_seq.append(next_button)
                buttons_pressed += 1
                last_button = next_button
                levels_seq.append(next_button)

            if pd.notna(next_level):
                levels_seq.append(next_level)
        
        if end_time is None: # Happens for the last Input before end of log
            end_time = df.iloc[-1]['timestamp'] # stayed until end of log in this mode
            exit_mode = 'End of Log'
            print(f'{start_time} - {end_time} = {(end_time - start_time).total_seconds()}') ###

        duration = (end_time - start_time).total_seconds()
        total_time += duration
        if printout:
            print(f"  ↩ Exited {mode_name} with '{exit_mode}' at {end_time}")
            print(f"    Buttons pressed: {buttons_pressed}")
            print(f"    Duration: {duration:.2f} seconds\n")

        mode_data.append({
            'timestamp': start_time,
            'buttons_pressed': buttons_pressed,
            'exit': exit_mode,
            'buttons_sequence': buttons_seq,
            'levels': levels_seq,
            'duration': duration,
            'turned_off': turnedoff_time,
        })

    if printout:
        print("-" * 60)
        print(f"Total {mode_button} presses: {mode_count}")
        print(f"Total time in {mode_name}: {total_time:.2f} seconds")
    
    return pd.DataFrame(mode_data)

def track_simple_mode(df, mode_button, mode_name, extra_flags=[], stay_buttons=[], comes_from_table_buttons=[], printout=False):
    """
    Simplified tracker for modes without selction options like Home, Drive, or Table.
    Input:
        df: df from get_informative_df()
    Output:
        DataFrame with:
        - #TODO
    """

    df = df.sort_values('timestamp').reset_index(drop=True)
    total_time = 0.0
    mode_count = 0
    mode_data = []

    for i, row in df.iterrows():
        if row['button'] != mode_button:
            continue

        mode_count += 1
        start_time = row['timestamp']
        end_time = None
        flags = {flag: False for flag in extra_flags}
        stopped = False
        turnedoff_time = 0

        if printout:
            print(f"[{start_time}] Activated {mode_name}")

        for j in range(i + 1, len(df)):
            next_row = df.iloc[j]

            # Handle system restart
            if next_row['startup']:
                end_time = df.iloc[j - 1]['timestamp'] # stayed until the last log time [j-1] before system restarted in this mode
                next_action = 'startup'
                if printout:
                    print(f"  → System restarted during {mode_name}")
                turnedoff_time = (df.iloc[j]['timestamp'] - df.iloc[j - 1]['timestamp']).total_seconds()
                break

            next_button = next_row['button']
            if pd.isna(next_button):
                continue

            if next_button in ['stop', 'GUI STOP pitchNegative']:
                stopped = True
                if printout:
                    print(f"  → Stopping action")

            elif next_button in comes_from_table_buttons:
                if 'from_table_mode' in flags:
                    flags['from_table_mode'] = True
                    if printout:
                        print(f"  → Comes from Table Mode")

            elif next_button in stay_buttons:
                if 'stayed' in flags:
                    flags['stayed'] = True
                    if printout:
                        print(f"  → Staying in Table Mode")

            else:
                end_time = next_row['timestamp'] # stayed until next button press in this mode
                next_action = next_button
                if printout:
                    print(f"  → Continued with '{next_button}' at {end_time}")
                break
        
        if end_time is None: # Happens for the last Input before end of log
            end_time = df.iloc[-1]['timestamp'] # stayed until end of log in this mode
            next_action = 'End of Log'
            print(f'{start_time} - {end_time} = {(end_time - start_time).total_seconds()}')

        duration = (end_time - start_time).total_seconds()
        total_time += duration

        if printout:
            print(f"    Duration: {duration:.2f} seconds\n")

        mode_data.append({
            'timestamp': start_time,
            'duration': duration,
            'stopped': stopped,
            **flags,
            'next': next_action,
            'turned_off': turnedoff_time,
        })

    if printout:
        print("-" * 60)
        print(f"Total {mode_button} presses: {mode_count}")
        print(f"Total time in {mode_name}: {total_time:.2f} seconds")

    return pd.DataFrame(mode_data)


# ================================
# Specific Mode Wrappers
# ================================
def get_manual(informative_df, printout=False):
    return track_mode(informative_df, mode_button='ManualInput', mode_name='Manual Mode', printout=printout)

def get_tasks(informative_df, printout=False):
    return track_mode(informative_df, mode_button='TasksInput', mode_name='Tasks Mode', printout=printout)

def get_fast(informative_df, printout=False):
    return track_mode(informative_df, mode_button='FastNavigationInput', mode_name='Fast Mode', printout=printout)

def get_settings(informative_df, printout=False):
    return track_mode(informative_df, mode_button='SettingsInput', mode_name='Settings', printout=printout)

def get_home(informative_df, printout=False):
    return track_simple_mode(
        informative_df,
        mode_button='HomeInput',
        mode_name='Home Mode',
        extra_flags=['from_table_mode'],
        comes_from_table_buttons=['yes', 'no'],
        printout=printout
    )

def get_drive(informative_df, printout=False):
    return track_simple_mode(
        informative_df,
        mode_button='DriveInput',
        mode_name='Drive Mode',
        printout=printout
    )

def get_table(informative_df, printout=False):
    return track_simple_mode(
        informative_df,
        mode_button='TableAccessInput',
        mode_name='Table Mode',
        extra_flags=['stayed'],
        stay_buttons=['no'],
        printout=printout
    )


# ================================
# Main Logic
# ================================
def main():
    current_dir = os.getcwd()
    base_path = os.path.abspath(os.path.join(current_dir, "data"))
    empty_files_counter = 0
    summary_list = []

    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        filename = f"EverydayGUI_log_{folder_name}.log"
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            print(f"The log file {folder_name} is empty.")
            empty_files_counter += 1
            continue

        with open(file_path, "r") as file:
            lines = file.readlines()

        print(f"Processing log: {folder_name}")
        df = get_informative_df(lines)

        # Track all modes
        modes = {
            'manualMode': get_manual(df),
            'tasksMode': get_tasks(df),
            'fastMode': get_fast(df),
            'settingsMode': get_settings(df),
            'homeMode': get_home(df),
            'driveMode': get_drive(df),
            'tableMode': get_table(df)
        }

        save_mode_csvs(modes, folder_path, folder_name)

        summary_list.append({
            'Date': folder_name,
            'Time in Manual Mode': get_total_duration(modes['manualMode']),
            'Time in Tasks Mode': get_total_duration(modes['tasksMode']),
            'Time in Fast Navigation Mode': get_total_duration(modes['fastMode']),
            'Time in Settings': get_total_duration(modes['settingsMode']),
            'Time in Home Mode': get_total_duration(modes['homeMode']),
            'Time in Drive Mode': get_total_duration(modes['driveMode']),
            'Time in Table Mode': get_total_duration(modes['tableMode']),
            'Total Log Time': get_total_log_time(df),
            'Number of Startups': startup_counter(df, folder_name)
        })

    # Print and export summary
    print(f"\nNumber of empty files: {empty_files_counter}")
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(base_path, "summary.csv"), index=False)
    print("\nSummary of all logs:")
    print(summary_df)

# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    main()