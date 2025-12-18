import pandas as pd
import re
import ast

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('expand_frame_repr', False)

STOP_EXITS = ['stopped (no)', 'GUI stop', 'head stop', 'stopped (done)', 'stopped (back)', 'manual stop', 'startup']

# ─────────────────────────────────────────────────────────
def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return val

# ─────────────────────────────────────────────────────────
# Stops
def get_stops_active(date, mode):
    file_path = f'data/{date}/{date}_{mode}Mode.csv'
    gui_stop = 0
    gui_moving_stop = 0
    head_stop = 0
    head_moving_stop = 0
    startup = 0
    manual_stop = 0

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {file_path}")
        return {}
    
    df['levels'] = df['levels'].apply(safe_eval)
    
    # loop through each row
    for idx, row in df.iterrows():
        if row.get('exit') == 'GUI stop':
            gui_stop += 1
            levels_value = row.get('levels')
            if len(levels_value) >= 5:
                if levels_value[-4] in ['am bewegen...', 'bewege zur Trink-/Essposition...', 'bewege zur Abgabeposition...', 'moving...', 'moving to drinking position...', 'moving to dropoff position...']:
                    gui_moving_stop += 1
        elif row.get('exit') == 'head stop':
            head_stop += 1
            levels_value = row.get('levels')
            if len(levels_value) >= 4:
                if levels_value[-3] in ['am bewegen...', 'bewege zur Trink-/Essposition...', 'bewege zur Abgabeposition...', 'moving...', 'moving to drinking position...', 'moving to dropoff position...']:
                    head_moving_stop += 1
        elif row.get('exit') == 'startup':
            startup += 1
            print(f'Startup in {mode} Mode, check for reason {row.get('levels')}')
        elif row.get('exit') == 'manual stop':
            manual_stop += 1


    return {
        f'{mode}: GUI Stop': gui_stop,
        f'{mode}: GUI Stop Moving': gui_moving_stop,
        f'{mode}: Head Stop': head_stop,
        f'{mode}: Head Stop Moving': head_moving_stop,
        f'{mode}: Manual Stop': manual_stop,
        f'{mode}: Startup': startup,
    }

def get_stops_inactive(date, mode):
    file_path = f'data/{date}/{date}_{mode}Mode.csv'

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {file_path}")
        return {}
    
    total_rows = len(df)

    # Count how many True in 'stopped'
    true_stopped = df['stopped'].sum()   # works because True=1, False=0

    return {
        f'{mode}: total': total_rows,
        f'{mode}: stopped': true_stopped,
    }

def get_stops(date):
    return {
        **get_stops_inactive(date, 'drive'),
        **get_stops_inactive(date, 'home'),
        **get_stops_inactive(date, 'table'),
        **get_stops_active(date, 'tasks'),
        **get_stops_active(date, 'fast'),
    }

# ─────────────────────────────────────────────────────────
# Restart
def get_restart_analysis(date):
    df = pd.read_csv(f'data/{date}/{date}_Startups.csv')
    # Startlogs and End logs
    # load settings for the startlogs, but it is not looked at in the query (can also be taken out)
    
    restart_s = []
    shutdown_s = []

    df_summary = pd.read_csv('data/summary.csv')
    total_restarts = df_summary.loc[df_summary['Date'] == date, 'Number of Startups'].values[0] # searches for 'load settings' in the log

    try:
        df_settings = pd.read_csv(f'data/{date}/{date}_settingsMode.csv')

        df_settings['levels'] = df_settings['levels'].apply(safe_eval)

        mask = df_settings['levels'].apply(lambda x: isinstance(x, (list, tuple)) and x[-1] == 'restart')
        restart_s.extend(df_settings.loc[mask, 'turned_off'].tolist())
        
        mask2 = df_settings['levels'].apply(lambda x: isinstance(x, (list, tuple)) and x[-1] == 'shutdown')
        shutdown_s.extend(df_settings.loc[mask2, 'turned_off'].tolist())
    
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: settings_file")

    a = 900 # (15min)

    if len([x for x in df.loc[df['mode'] != 'StartLog', 'turned_off'] if x < a and x != 0]) + len([x for x in df.loc[df['mode'] != 'StartLog', 'turned_off'] if x >= a or x == 0]) != total_restarts:
        print('achtung')
    return {
        'settings restart: restarts': len([x for x in restart_s if x < a and x != 0]), # in settings you also have End of log (time 0) init (not searched for startup)
        'settings shutdown: restarts': len([x for x in shutdown_s if x < a and x != 0]),

        'settings restart: shutdowns': len([x for x in restart_s if x >= a or x == 0]), # in settings you also have End of log (time 0) init (not searched for startup)
        'settings shutdown: shutdowns': len([x for x in shutdown_s if x >= a or x == 0]), 

        'total: restarts': len([x for x in df.loc[df['mode'] != 'StartLog', 'turned_off'] if x < a and x != 0]), # End of log (time 0)
        'total: shutdowns': len([x for x in df.loc[df['mode'] != 'StartLog', 'turned_off'] if x >= a or x == 0]), # End of log (time 0)

        'settings: restarts': len([x for x in df.loc[df['mode'] == 'Settings', 'turned_off'] if x < a and x != 0]), # End of log (time 0)
        'fast: restarts': len([x for x in df.loc[df['mode'] == 'FastNavigation', 'turned_off'] if x < a and x != 0]),
        'tasks: restarts': len([x for x in df.loc[df['mode'] == 'Tasks', 'turned_off'] if x < a and x != 0]),
        'manual: restarts': len([x for x in df.loc[df['mode'] == 'Manual', 'turned_off'] if x < a and x != 0]),
        'home: restarts': len([x for x in df.loc[df['mode'] == 'Home', 'turned_off'] if x < a and x != 0]),
        'table: restarts': len([x for x in df.loc[df['mode'] == 'TableAccess', 'turned_off'] if x < a and x != 0]),
        'drive: restarts': len([x for x in df.loc[df['mode'] == 'Drive', 'turned_off'] if x < a and x != 0]),
        'homepage: restarts': len([x for x in df.loc[df['mode'] == 'Homepage', 'turned_off'] if x < a and x != 0]),

        'settings: shutdowns': len([x for x in df.loc[df['mode'] == 'Settings', 'turned_off'] if x >= a or x == 0]),
        'fast: shutdowns': len([x for x in df.loc[df['mode'] == 'FastNavigation', 'turned_off'] if x >= a or x == 0]),
        'tasks: shutdowns': len([x for x in df.loc[df['mode'] == 'Tasks', 'turned_off'] if x >= a or x == 0]),
        'manual: shutdowns': len([x for x in df.loc[df['mode'] == 'Manual', 'turned_off'] if x >= a or x == 0]),
        'home: shutdowns': len([x for x in df.loc[df['mode'] == 'Home', 'turned_off'] if x >= a or x == 0]),
        'table: shutdowns': len([x for x in df.loc[df['mode'] == 'TableAccess', 'turned_off'] if x >= a or x == 0]),
        'drive: shutdowns': len([x for x in df.loc[df['mode'] == 'Drive', 'turned_off'] if x >= a or x == 0]),
        'homepage: shutdowns': len([x for x in df.loc[df['mode'] == 'Drive', 'turned_off'] if x >= a or x == 0]),

        'total: restarts + shutdowns': total_restarts, # startlog counted but not end log so cancel each other out
    }


# ─────────────────────────────────────────────────────────
# Gripper
def get_gripper_orientation(date, mode):
    """
    mode: 'fast' or 'tasks'
    Produces keys:
        fast  → "gripper: up"
        tasks → "tasks gripper: up"
    """
    assert mode in ("fast", "tasks"), "mode must be 'fast' or 'tasks'"

    prefix = "gripper" if mode == "fast" else "tasks gripper"
    path = f"data/{date}/{date}_{mode}Mode.csv"

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {path}")
        # Empty dataset → all None
        return {}

    df['buttons_sequence'] = df['buttons_sequence'].apply(safe_eval)

    stopped_exits = {
        "GUI stop", "head stop", "stopped (no)",
        "stopped (done)", "stopped (back)", "manual stop"
    }

    button_map = {
        'Gripper Orientation Selection GripperOrientations.Up': 'up',
        'Gripper Orientation Selection GripperOrientations.Down': 'down',
        'Gripper Orientation Selection GripperOrientations.Center': 'center',
        'Gripper Orientation Selection GripperOrientations.Left': 'left',
        'Gripper Orientation Selection GripperOrientations.Right': 'right',
        'back': 'back'
    }

    # Counters
    orientation_counts = {key: 0 for key in button_map.values()}
    orientation_stopped = {label: 0 for label in button_map.values() if label != 'back'}

    # -------------------------
    # MODE-SPECIFIC EXTRACTION
    # -------------------------
    def extract_label(seq):
        """Return appropriate label or None based on mode rules."""

        if not isinstance(seq, list) or not seq:
            return None

        # FAST MODE: only first item
        if mode == "fast":
            return button_map.get(seq[0], None)

        # TASKS MODE: first button AFTER 'fastNavigation'
        if mode == "tasks":
            for i in range(len(seq)):
                if seq[i] == 'fastNavigation':
                    return button_map.get(seq[i+1], None)

    # -------------------------
    # PROCESS ROWS
    # -------------------------
    for _, row in df.iterrows():
        label = extract_label(row["buttons_sequence"])
        if label:
            orientation_counts[label] += 1
            if label != "back" and row.get("exit") in stopped_exits:
                orientation_stopped[label] += 1

    # Build result with prefix applied to labels
    result = {}
    for label in button_map.values():
        result[f"{prefix}: {label}"] = orientation_counts[label]
        if label != 'back':
            result[f"{prefix}: {label} stopped"] = orientation_stopped[label]
    result[f"{prefix}: total rows"] = len(df)

    return result

# Tasks
def get_prefered_mode(date):
    file_path = f'data/{date}/{date}_tasksMode.csv'

    grab = {
        'grab: onFloor': 0,
        'grab: onFloor finished': 0,
        'grab: onFloor stopped': 0,

        'grab: onTable': 0,
        'grab: onTable finished': 0,
        'grab: onTable stopped': 0,

        'grab: currentPose': 0,
        'grab: currentPose finished': 0,
        'grab: currentPose stopped': 0,

        'grab: fastNavigation': 0,
        'grab: fastNavigation finished': 0,
        'grab: fastNavigation stopped': 0,

        'grab: back': 0, # wird als stop gezählt enthält, auch GUI STOP pitchNegative

        'grab end: drink': 0,
        'grab end: drink finished': 0,
        'grab end: drink stopped': 0,

        'grab end: toTable': 0,
        'grab end: toTable finished': 0,
        'grab end: toTable stopped': 0,

        'grab end: fastNavigation': 0,
        'grab end: fastNavigation finished': 0,
        'grab end: fastNavigation stopped': 0,

        'grab end: toUserDefinedPose': 0,
        'grab end: toUserDefinedPose finished': 0,
        'grab end: toUserDefinedPose stopped': 0,

        'grab end: done': 0,

        'total: grab': 0,
        'total: grab finished': 0,
        'total: grab stopped': 0,
        
    }

    push = {
        'push: onWall': 0,
        'push: onWall finished': 0,
        'push: onWall stopped': 0,

        'push: onTable': 0,
        'push: onTable finished': 0,
        'push: onTable stopped': 0,

        'push: currentPose': 0,
        'push: currentPose finished': 0,
        'push: currentPose stopped': 0,

        'push: fastNavigation': 0,
        'push: fastNavigation finished': 0,
        'push: fastNavigation stopped': 0,

        'push: back': 0, # wird als stop gezählt, enthält auch GUI STOP pitchNegative

        'total: push': 0,
        'total: push finished': 0,
        'total: push stopped': 0,

    }

    next_mode = {
        'next: toTable': 0,
        'next: toTable finished': 0,
        'next: toTable stopped': 0,

        'next: drink': 0,
        'next: drink finished': 0,
        'next: drink stopped': 0,

        'next: fastNavigation': 0,
        'next: fastNavigation finished': 0,
        'next: fastNavigation stopped': 0,

        'next: toUserDefinedPose': 0,
        'next: toUserDefinedPose finished': 0,
        'next: toUserDefinedPose stopped': 0,

        'next: back': 0, # wird als stop gezählt, enthält auch GUI STOP pitchNegative, done

        'total: next': 0,
        'total: next finished': 0,
        'total: next stopped': 0,
    }

    ends_with_stop = None
    ends_without_stop = None
    returned = 0

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {file_path}")
        return {}

    df['buttons_sequence'] = df['buttons_sequence'].apply(safe_eval)

    for _, row in df.iterrows():
        seq = list(row['buttons_sequence'])
        iexit = row['exit']

        if len(seq) == 1:
            returned += 1
            continue # breaks row loop

        # Find last relevant action
        for idx in range(len(seq) - 1, -1, -1):
            action = seq[idx]

            if action in ['grab', 'push', 'next']:
                break
        if action == 'grab':
            grab[f'total: grab'] += 1
            
            if seq[idx+1] in ['back', 'GUI STOP pitchNegative']:
                grab['grab: back'] += 1
                grab['total: grab stopped'] += 1
                continue

            grab[f'grab: {seq[idx+1]}'] += 1

            if iexit in STOP_EXITS:
                grab[f'grab: {seq[idx+1]} stopped'] += 1
                grab[f'total: grab stopped'] += 1

            else: # ['finished (yes)', 'finished (done)', 'finished (back)', 'finished (fn)']
                grab[f'total: grab finished'] += 1
                grab[f'grab: {seq[idx+1]} finished'] += 1

            # Track what happened after the grab
            if iexit == 'finished (done)':
                grab['grab end: done'] += 1
                continue

            for j in range(idx+2, len(seq)): # idx: grab, idx+1: grap_seq, idx+2: danach                          
                if seq[j] in ['drink','toTable','fastNavigation','toUserDefinedPose']:
                    grab[f'grab end: {seq[j]}'] += 1
                    if iexit in STOP_EXITS:
                        grab[f'grab end: {seq[j]} stopped'] += 1
                    else:
                        grab[f'grab end: {seq[j]} finished'] += 1
                    continue # breaks j loop #TODO what to do when you go back 
                    
        elif action == 'push':
            push[f'total: push'] += 1

            if seq[idx+1] in ['back', 'GUI STOP pitchNegative']: # counts as stop
                push['push: back'] += 1
                push['total: push stopped'] += 1
                continue
                
            push[f'push: {seq[idx+1]}'] += 1

            if iexit in STOP_EXITS:
                push[f'push: {seq[idx+1]} stopped'] += 1
                push[f'total: push stopped'] += 1

            else: # ['finished (yes)', 'finished (done)', 'finished (back)', 'finished (fn)']
                push[f'push: {seq[idx+1]} finished'] += 1
                push[f'total: push finished'] += 1


        else: # action == 'next':
            next_mode[f'total: next'] += 1

            if seq[idx+1] in ['back', 'GUI STOP pitchNegative', 'done']: # counts as stop
                next_mode['next: back'] += 1
                next_mode['total: next stopped'] += 1
                continue
                
            next_mode[f'next: {seq[idx+1]}'] += 1

            if iexit in STOP_EXITS:
                next_mode[f'next: {seq[idx+1]} stopped'] += 1
                next_mode[f'total: next stopped'] += 1
                
            else: # ['finished (yes)', 'finished (done)', 'finished (back)', 'finished (fn)']
                next_mode[f'next: {seq[idx+1]} finished'] += 1
                next_mode[f'total: next finished'] += 1


    # Check sequences that end with stop/estop vs others
    
    ends_with_stop = df['buttons_sequence'].apply(lambda x: x and x[-1] in {'stop', 'estop'}).sum()
    ends_without_stop = df['buttons_sequence'].apply(lambda x: x and x[-1] not in {'stop', 'estop'}).sum()

    return {
        **grab,
        **push,
        **next_mode,
        'overall tasks returned': returned,
        'overall tasks': len(df),
        'overall tasks finished': ends_without_stop,
        'overall tasks stopped': ends_with_stop
        }

# ─────────────────────────────────────────────────────────
# Vision
def get_detection_analysis(seqs):

    options = 0
    more_opt = 0
    not_first = 0
    total_retries = 0
    flip = 0

    grab = 0
    push = 0
    grab_retries = 0
    push_retries = 0
    grab_unable = 0
    push_unable = 0
    grab_stopped = 0
    push_stopped = 0
    grab_good = 0
    push_good = 0
    grab_bad = 0
    push_bad = 0
    grab_stopped_after = 0
    push_stopped_after = 0
    grab_mult_opt = 0
    push_mult_opt = 0
    grab_not_first = 0
    push_not_first = 0

    for seq in seqs:
        flip_int = 0

        mode = seq.pop(0)
        mode_count = 0
        unable = 0
        stopped = 0
        good = 0
        bad = 0
        stopped_after = 0
        more_opt = 0
        not_first = 0

        while True:
            # position a

            if seq[0] in ['DrÃ¼cke auf das gewÃ¼nschte Objekt', 'Select Point on image']:
                unable += 1
                for i in range(len(seq)):
                    if seq[i] == 'ok':
                        total_retries += 1
                        mode_count += 1
                        seq = seq[i+3:] # remove until it goes again into segmentation
                        break # i loop
                else:
                    stopped_after += 1
                    break  # done with this sequence

                continue # restart from position a

            if seq[0] == 'GUI STOP pitchNegative':
                stopped += 1
                break # done with this sequence


            match_de = re.match(r"Ist die Detektion okay\? Option:\((\d+) / (\d+)\)", seq[0])
            match_en = re.match(r"Detection okay\? Option:\((\d+) / (\d+)\)", seq[0])
            match = match_de or match_en  # pick the one that matched

            if match:
                cur_opt = int(match.group(1))  # Extracted current option number
                tot_opt = int(match.group(2))  # Extracted total options

                if seq[2] == 'yes': # continued in the tasks mode
                    good += 1
                    options += tot_opt
                    if tot_opt != 1:
                        more_opt += 1

                    if cur_opt != 1:
                        not_first += 1
                    break  # done with this sequence

                elif seq[2] == 'no':
                    bad += 1

                    for i in range(len(seq)):
                        if seq[i] == 'ok':
                            total_retries += 1
                            mode_count += 1
                            seq = seq[i+3:] # remove until it goes again into segmentation and restart from position a
                            break # i loop
                    else:
                        stopped_after += 1
                        break # done with this sequence

                    continue # restart from position

                elif seq[2] in ['>', '<']:
                    seq = seq[3:]
                    continue

                elif seq[2] =='flip':
                    flip_int += 1
                    seq = seq[3:]
                    continue

                elif seq[2] == 'GUI STOP pitchNegative':
                    stopped += 1
                    break # done with this sequence
                else:
                    print('issue at detection')
                    break # done with this sequence
            else:
                print("issue at detection")
                break # done with this sequence
        
        if mode == 'grab':
            grab += (mode_count+1) # +1 for the seq, mode_count for the retries
            grab_retries += mode_count
            grab_unable += unable
            grab_stopped += stopped
            grab_good += good
            grab_bad += bad
            grab_stopped_after += stopped_after
            grab_mult_opt += more_opt
            grab_not_first += not_first
        elif mode == 'push':
            push += (mode_count+1)
            push_retries += mode_count
            push_unable += unable
            push_stopped += stopped
            push_good += good
            push_bad += bad
            push_stopped_after += stopped_after
            push_mult_opt += more_opt
            push_not_first += not_first
        else:
            print("issue at detection")

        # seq done
        if flip_int % 2 != 0:
            flip += 1

    return {
        'detection: options': options,

        'detection: grab unable': grab_unable,
        'detection: grab bad': grab_bad,
        'detection: grab stopped': grab_stopped,
        'detection: grab stopped after': grab_stopped_after,
        'detection: grab good': grab_good,
        'detection: grab retries': grab_retries,
        'detection: grab': grab, # grab + grab_retires = good + stopped + unable + bad
        'detection: grab multiple options': grab_mult_opt,
        'detection: grab not first choice': grab_not_first,
        'detection: gripper changed': flip,

        'detection: push unable': push_unable,
        'detection: push bad': push_bad,
        'detection: push stopped': push_stopped,
        'detection: push stopped after': push_stopped_after,
        'detection: push good': push_good,
        'detection: push retries': push_retries,
        'detection: push': push,
        'detection: push multiple options': push_mult_opt,
        'detection: push not first choice': push_not_first,

        'detection: total detections': len(seqs) + total_retries, # = good + stopped + unable + bad
        'detection: total': len(seqs), # = good + stopped, how often it appear in a task # maybe not important
    }

def get_control_analysis(seqs):

    stopped = 0

    push = 0
    push_yes = 0
    push_yes_first = 0
    push_no = 0
    push_stopped = 0
    push_stopped_first = 0

    grab = 0
    grab_yes = 0
    grab_yes_first = 0
    grab_no = 0
    grab_no_first = 0
    grab_gripper = 0
    grab_stopped = 0
    grab_stopped_man = 0
    grab_stopped_first = 0
    grab_closer = 0
    grab_alt = 0

    grab2 = 0
    grab_yes2 = 0
    grab_no2 = 0
    grab_gripper2 = 0
    grab_stopped2 = 0
    grab_stopped2_man = 0
    grab_alt2 = 0

    grab_start = 0
    push_start = 0

    grab_fin = 0
    push_fin = 0
    
    for seq in seqs:

        # in which mode started, so that it also counts the stopped 
        mode = seq.pop(0)
        if mode == 'grab':
            grab_start += 1 
        elif mode == 'push':
            push_start += 1
        else:
            print("issue at control")

        first = True

        if seq[0] in ['GUI STOP pitchNegative', 'stop']:
            stopped += 1
            continue

        if seq[2] in ['BerÃ¼hrt?', 'Touched?']:
            while True:
                push += 1
                ### position c
                if seq[4] == 'yes':
                    push_yes += 1
                    push_fin += 1
                    if first:
                        push_yes_first += 1
                    else:
                        first = False
                    ## goes to position end
                    break # done with this sequence 
                elif seq[4] == 'no':
                    push_no += 1
                    seq = seq[5:]
                    first = False
                    ## goes to position c
                elif seq[4] in ['GUI STOP pitchNegative', 'stop']:
                    push_stopped += 1
                    if first:
                        push_stopped_first += 1
                    ## goes to position end
                    break # done with this sequence 
                else:
                    print('issue in grab', seq[4])
                    ## goes to position end
                    break # done with this sequence

            continue

        while True: # Bereit zu greifen?
            ### position a
            grab += 1
            if seq[2] == 'yes':
                if first:
                    grab_yes_first += 1
                else:
                    first = False
                grab_yes += 1
                seq = seq[3:]
                ## goes to position b
            elif seq[2] == 'no':
                if first:
                    grab_no_first += 1
                else:
                    first = False
                grab_no += 1
                
                for i in range(5, len(seq)): # 3: manual, 4: manualV1
                    if seq[i] == 'back':
                        manual1.append(seq[5:i])
                        if seq[i+1] == 'back': #issue double back
                            seq = seq[i+2:]
                        else:
                            seq = seq[i+1:]
                        ## goes to position b 
                        break # i loop
                    elif seq[i] in ['Gripper Open', 'Gripper Close']:
                        grab_gripper += 1
                    grab_alt += 1 # does not count back but gripper as well
                else:
                    grab_stopped_man += 1
                    grab_alt -= 1 # remove estop
                    manual1.append(seq[5:])
                    ## goes to position end
                    break # done with this sequence
                 
            elif seq[2] == 'closer':
                grab_closer += 1
                first = False
                # issue, do to probably fast pressing of closer, 
                for i in range(3, len(seq)):
                    if seq[i] in ['yes', 'no', 'GUI STOP pitchNegative', 'stop']:
                        seq = seq[i-2:]
                        break # i loop
                    elif seq[i] == 'closer':
                        grab_closer += 1
                        grab += 1 # because it is skipped
                ## goes to position a
                continue # restart while loop
            elif seq[2] in ['GUI STOP pitchNegative', 'stop']:
                grab_stopped += 1
                if first:
                        grab_stopped_first += 1
                ## goes to position end
                break # done with this sequence
            else:
                print('issue in grab', seq, seq[2])
                ## goes to position end
                break # done with this sequence

            ### position b
            # Greifer geschlossen?
            grab2 += 1
            if seq[2] == 'yes':
                grab_yes2 += 1
                grab_fin += 1
                ## goes to position end
            elif seq[2] == 'no':
                grab_no2 += 1
                
                for i in range(5, len(seq)): # 3: manual, 4: manualV1
                    if seq[i] == 'back': # Greifer geschlossen
                        manual2.append(seq[5:i])
                        grab_fin += 1
                        ## goes to position end
                        break # i loop
                    elif seq[i] in ['Gripper Open', 'Gripper Close']:
                        grab_gripper2 += 1 # jedesmal
                    grab_alt2 += 1
                else:
                    grab_stopped2_man += 1
                    grab_alt2 -= 1 # remove estop
                    manual2.append(seq[5:])
                    ## goes to position end
                    break # done with this sequence
            elif seq[2] in ['GUI STOP pitchNegative', 'stop']:
                grab_stopped2 += 1
                ## goes to position end
                break # done with this sequence
            else:
                print('issue in grab', seq, seq[2])
                ## goes to position end
                break # done with this sequence

            # position end
            break
    
    return {
        # Berührt?        
        'control: push good': push_yes,
        'control: push good first': push_yes_first,
        'control: push bad': push_no, # repeats the loop
        'control: push stopped': push_stopped, # no good reason to press stop, can also be considered as good
        'control: push stopped first': push_stopped_first,
        'control: push total': push, # = yes + no + stopped

        # Bereit zu greifen?
        'control: grab good': grab_yes,
        'control: grab good first': grab_yes_first,
        'control: grab bad': grab_no,
        'control: grab bad first': grab_no_first,
        'control: grab altered': grab_alt,
        'control: grab gripper': grab_gripper, # part of grab_no, voreilig Roboter macht es erst danach
        'control: grab stopped': grab_stopped,
        'control: grab stopped manual': grab_stopped_man,
        'control: grab stopped first': grab_stopped_first,
        'control: grab closer': grab_closer,
        'control: grab total': grab, # = yes + no + stopped + closer

        # Greifer geschlossen?
        'control: grab end good': grab_yes2,
        'control: grab end bad': grab_no2,
        'control: grab end altered': grab_alt2,
        'control: grab end gripper': grab_gripper2, # part of grab_no, korrigieren des Roboters
        'control: grab end stopped': grab_stopped2,
        'control: grab end stopped manual': grab_stopped2_man,
        'control: grab end total': grab2, # = yes + no + stopped

        'control: stopped': stopped,
        
        'control: total': len(seqs),
        'control: grab start': grab_start,
        'control: push start': push_start,
        'control: grab finished': grab_fin,
        'control: push finished': push_fin,
    }

from collections import Counter

manual1 = []
manual2 = []

def get_tasks_manual_analysis(seqs):

    counts = Counter(item for seq in seqs for item in seq)

    normalized = Counter()
    for key, count in counts.items():
        norm_key = key.replace("Stepwise ", "").replace("Continous ", "")
        normalized[norm_key] += count

    return normalized

def get_manual_analysis(date):
    """
    Loads the dataframe, extracts button sequences, cleans them
    and returns normalized counts for each button action.
    """

    file_path = f'data/{date}/{date}_manualMode.csv'

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {file_path}")
        return {}

    series = df["buttons_sequence"].apply(safe_eval)

    # ✔ get normalized counts
    counts = get_tasks_manual_analysis(series)

    # ✔ return all required fields (defaulting to 0 if missing)
    return {
        'manual: down': counts.get("down", 0),
        'manual: up': counts.get("up", 0),
        'manual: forward': counts.get("forward", 0),
        'manual: backward': counts.get("backward", 0),
        'manual: left': counts.get("left", 0),
        'manual: right': counts.get("right", 0),
        'manual: Gripper Close': counts.get("Gripper Close", 0),
        'manual: Gripper Open': counts.get("Gripper Open", 0),
        'manual: rollNegative': counts.get("rollNegative", 0),
        'manual: rollPositive': counts.get("rollPositive", 0),
        'manual: yawNegative': counts.get("yawNegative", 0),
        'manual: yawPositive': counts.get("yawPositive", 0),
        'manual: pitchNegative': counts.get("pitchNegative", 0),
        'manual: pitchPositive': counts.get("pitchPositive", 0),
    }

def get_vision_analysis(date):
    file_path = f'data/{date}/{date}_tasksMode.csv'

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"[INFO] File is empty: {file_path}")
        return {}

    df['levels'] = df['levels'].apply(safe_eval)

    detection_seqs = []
    control_seqs = []

    mode = ''

    for seq in df['levels']:
        for i in range(len(seq)):
            if seq[i] == 'ok': # i-2: 'DrÃ¼cke auf das gewÃ¼nschte Objekt', i-1: 'prompt', i+1: 'Bitte warten...', i+2: 'prompt'
                for l in range(i - 1, -1, -1): # go backwards and look in which mode it is
                    if seq[l] == 'grab':
                        mode = 'grab'
                        break # l-loop
                    elif seq[l] == 'push':
                        mode = 'push'
                        break
                for j in range(i, len(seq)):
                    if seq[j] == 'yes':
                        detection_seqs.append([mode] + seq[i+3:j+3])
                        control_seqs.append([mode] + seq[j+3:])
                        break
                else: # when the for loop finishes without break
                    detection_seqs.append([mode] + seq[i+3:])
                break

    detection = get_detection_analysis(detection_seqs)
    control = get_control_analysis(control_seqs)

    return {
        **detection,
        **control
    }

def mode_count(date): #TODO
    t_stop = 0
    m_stop = 0
    f_stop = 0

    t_return = 0
    m_return = 0
    f_return = 0

    try:
        ta = pd.read_csv(f'data/{date}/{date}_tasksMode.csv')
        tasks = len(ta)
        ta['buttons_sequence'] = ta['buttons_sequence'].apply(safe_eval)
        for _, row in ta.iterrows():
            seq = list(row['buttons_sequence'])
            iexit = row['exit']
            
            if len(seq) == 1:
                t_return += 1

            if iexit in STOP_EXITS:
                t_stop += 1

    except pd.errors.EmptyDataError:
        tasks = 0
    
    try:
        ma = pd.read_csv(f'data/{date}/{date}_manualMode.csv')
        manual = len(ma)
        ma['buttons_sequence'] = ma['buttons_sequence'].apply(safe_eval)
        for _, row in ma.iterrows():
            seq = list(row['buttons_sequence'])
            iexit = row['exit']

            if len(seq) == 1:
                m_return += 1
                m_stop += 1 # because you can also go back with back

            elif iexit in ['estop']:
                m_stop += 1

    except pd.errors.EmptyDataError:
        manual = 0

    try:
        fa = pd.read_csv(f'data/{date}/{date}_fastMode.csv')
        fast = len(fa)
        fa['buttons_sequence'] = fa['buttons_sequence'].apply(safe_eval)
        for _, row in fa.iterrows():
            seq = list(row['buttons_sequence'])
            iexit = row['exit']

            if len(seq) == 1:
                f_return += 1

            if iexit in STOP_EXITS:
                f_stop += 1

    except pd.errors.EmptyDataError:
        fast = 0

    return {
        'tasks start': tasks,
        'tasks return': t_return,
        'tasks stop': t_stop,
        'manual start': manual,
        'manual return': m_return,
        'manual stop': m_stop,
        'fast start': fast,
        'fast return': f_return,
        'fast stop': f_stop,
    }

# ─────────────────────────────────────────────────────────
def main():
    df = pd.read_csv('data/summary.csv')

    results = []

    for date in df['Date']:
        print(f"[INFO] Analyzing date: {date}")
        results.append({
            'date': date,
            **get_stops(date),
            **get_restart_analysis(date),
            **get_gripper_orientation(date, 'fast'),
            **get_gripper_orientation(date, 'tasks'),
            **get_prefered_mode(date),
            **get_vision_analysis(date),
            **mode_count(date),
            **get_manual_analysis(date),
        })
    
    man1 = get_tasks_manual_analysis(manual1)
    man2 = get_tasks_manual_analysis(manual2)

    result = pd.DataFrame(results)
    result.to_csv( "data/analysis.csv", index=False)

    return man1, man2

if __name__ == "__main__":
    main()