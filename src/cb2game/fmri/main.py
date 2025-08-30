import logging
import json
import os
import argparse
import pygame
import time
import csv
from datetime import datetime, timedelta
from pynput import keyboard
import numpy as np
import pandas as pd

# Add filelock import with error handling
try:
    from filelock import FileLock
except ImportError:
    logging.error("filelock package not found. Please install it with: pip install filelock")
    raise

from cb2game.pyclient.remote_client import RemoteClient
from cb2game.pyclient.game_endpoint import Action
from cb2game.fmri.utils import OBJECTIVE, N_TRIALS, MAX_TRIAL_DURATION, MAX_RUN_DURATION, open_browser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
FONT_SIZE = 75
HOST = "http://localhost:8080"
LOBBY = "scenario-lobby"
MATERIALS_DIR = 'materials'
MAX_TARGETS = None
KEYBOARD = keyboard.Controller()

# CSV logging constants
CSV_FILE_PATH = 'behavioral_data.csv'
CSV_LOCK_PATH = 'behavioral_data.csv.lock'

# CSV column headers
CSV_HEADERS = [
    'subject_id',
    'run_number',
    'experiment_start_utc',
    'scenario_position',
    'condition_code',
    'condition_detail',
    'palindrome_half',
    'scenario_filename',
    'instructions_text',
    'cards_correct',
    'cards_incorrect',
    'incorrect_card_ids',
    'difficulty_rating',
    'trial_duration_seconds',
    'trial_completed_utc'
]


def condition_to_code(task_difficulty, linguistic_complexity):
    """Convert condition tuple to two-letter code"""
    setting = "H" if task_difficulty > 0 else "E"  # Hard/Easy Setting
    language = "H" if linguistic_complexity > 0 else "E"  # Hard/Easy Language
    return f"{setting}{language}"


def condition_to_detail(task_difficulty, linguistic_complexity):
    """Convert condition tuple to readable description"""
    setting = "Hard_Setting" if task_difficulty > 0 else "Easy_Setting"
    language = "Hard_Language" if linguistic_complexity > 0 else "Easy_Language"
    return f"{setting}_{language}"


def extract_game_performance(game_state, trial):
    """Extract performance metrics from the trial's enhanced tracking"""
    try:
        # Use the enhanced tracking from the Trial class
        performance_summary = trial.get_performance_summary()
        return performance_summary

    except Exception as e:
        logger.warning(f"Could not extract game performance: {e}")
        # Fallback to basic data
        return {
            'cards_correct': trial.targets_found,
            'cards_incorrect': 0,
            'incorrect_card_ids': [],
            'instructions_text': "Error extracting instruction",
            'all_instructions': [],
            'total_card_selections': 0
        }


def ensure_csv_exists():
    """Create CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        logger.info(f"Created new CSV file: {CSV_FILE_PATH}")


def write_trial_data_to_csv(trial_data):
    """
    Safely append trial data to the CSV file using file locking
    """
    try:
        # Ensure CSV exists
        ensure_csv_exists()

        # Use file locking to prevent concurrent writes
        with FileLock(CSV_LOCK_PATH, timeout=10):
            with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(trial_data)

        logger.info(
            f"Successfully wrote trial data to CSV: {trial_data['subject_id']}, position {trial_data['scenario_position']}")

    except Exception as e:
        logger.error(f"Failed to write trial data to CSV: {e}")
        # Fallback: write to backup file
        backup_file = f"behavioral_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(backup_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                writer.writerow(trial_data)
            logger.warning(f"Wrote trial data to backup file: {backup_file}")
        except Exception as backup_error:
            logger.error(f"Failed to write backup file: {backup_error}")


def get_condition_from_scenario_path(scenario_path):
    """Extract condition from scenario filename"""
    try:
        filename = os.path.basename(scenario_path)
        # Parse filename like "scenario_1_t1_l0.json"
        parts = filename.split('_')
        if len(parts) >= 4:
            task_difficulty = int(parts[2][1])  # Extract from 't1'
            linguistic_complexity = int(parts[3][1])  # Extract from 'l0'
            return task_difficulty, linguistic_complexity
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse condition from scenario path {scenario_path}: {e}")

    # Fallback - return default
    return 0, 0


def validate_scenario_files(materials_dir, run_set, conditions):
    """
    Validate that all required scenario files exist for the given conditions.
    The validation is slight overkill, but the function also creates
    the list of relevant scenario jsons for palindrome construction.

    Args:
        materials_dir: Base materials directory path
        run_set: runset directory name, like 'runset_A'
        conditions: List of (task_difficulty, linguistic_complexity) tuples passed in fmri args

    Returns:
        tuple: (all_files_exist: bool, missing_files: list, available_files: dict)
    """
    runset_path = os.path.join(materials_dir, run_set)

    if not os.path.exists(runset_path):
        logger.error(f"Runset directory does not exist: {runset_path}")
        return False, [runset_path], {}

    # Get all available scenario files
    available_files = {}
    missing_files = []

    try:
        all_files = os.listdir(runset_path)
        scenario_files = [f for f in all_files if f.startswith('scenario_') and f.endswith('.json')]

        # Parse available materials files for scenarios in runset
        for filename in scenario_files:
            try:
                parts = filename.split('_')
                if len(parts) >= 4:
                    scenario_id = int(parts[1])
                    task_difficulty = int(parts[2][1])  # Extract number from 't0', 't1', etc.
                    linguistic_complexity = int(parts[3][1])  # Extract number from 'l0', 'l1', etc.

                    condition = (task_difficulty, linguistic_complexity)
                    if condition not in available_files:
                        available_files[condition] = []
                    available_files[condition].append({
                        'filename': filename,
                        'scenario_id': scenario_id,
                        'full_path': os.path.join(runset_path, filename)
                    })
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse scenario filename: {filename} - {e}")

        # Check if all required conditions have files
        all_files_exist = True
        for condition in conditions:
            if condition not in available_files or len(available_files[condition]) == 0:
                missing_files.append(
                    f"No scenarios found for condition t{condition[0]}_l{condition[1]} in {runset_path}")
                all_files_exist = False
            else:
                # Sort scenarios by ID for consistent ordering
                available_files[condition].sort(key=lambda x: x['scenario_id'])
                logger.info(
                    f"Found {len(available_files[condition])} scenarios for condition t{condition[0]}_l{condition[1]}")

        return all_files_exist, missing_files, available_files

    except OSError as e:
        logger.error(f"Error accessing directory {runset_path}: {e}")
        return False, [str(e)], {}


class Trial:
    def __init__(
            self,
            scenario_path,
            state=None,
            subject_kwargs=None,
            display=None,
            max_targets=MAX_TARGETS,
            max_trial_duration=MAX_TRIAL_DURATION
    ):
        logger.info(f"Loading scenario...")
        self.scenario_path = scenario_path
        self.state = state
        self.subject_kwargs = subject_kwargs
        if display is None:
            self.display = Display()
        else:
            self.display = display
        self.max_targets = max_targets
        self.max_trial_duration = max_trial_duration
        self.scenario_data = self.load_scenario_data(
            scenario_file=self.scenario_path,
            subject_kwargs=self.subject_kwargs
        )

        self.over = False
        self.success = False
        self.interrupted = False
        self.start_time = None
        self.target_found_times = []
        self.targets_found = 0
        self.n_moves = 0
        self.duration = None

        # Enhanced tracking for CSV logging
        self.card_selections = []  # Track all card selections
        self.incorrect_selections = []  # Track incorrect card IDs
        self.current_instruction = None  # Track current instruction text
        self.all_instructions = []  # Track all instructions seen

    def load_scenario_data(
            self,
            scenario_file,
            subject_kwargs=None
    ):
        if subject_kwargs is None:
            subject_kwargs = {}
        with open(scenario_file, "r") as f:
            scenario_data_json = f.read()
        scenario_data = json.loads(scenario_data_json)
        if "subject" not in scenario_data:
            scenario_data["subject"] = {}
        scenario_data["subject"].update(subject_kwargs)
        if self.max_trial_duration:
            scenario_data['duration_s'] = self.max_trial_duration
        else:
            scenario_data['duration_s'] = 3600
        if self.max_targets:
            scenario_data['target_card_ids'] = scenario_data['target_card_ids'][:self.max_targets]
            scenario_data['objectives'] = scenario_data['objectives'][:self.max_targets]

        return scenario_data

    def update_from_state(
            self,
            scenario_data
    ):
        state = self.state.to_dict()
        scenario_data['prop_update']['props'] = state['props']
        scenario_data['turn_state']['score'] = state['turn_state']['score']
        scenario_data['target_card_ids'] = scenario_data['target_card_ids'][state['turn_state']['score']:]
        scenario_data['objectives'] = scenario_data['objectives'][state['turn_state']['score']:]
        location = state['actors'][0]['location']
        rotation_degrees = state['actors'][0]['rotation_degrees']
        for actor in scenario_data['actor_state']['actors']:
            if actor['actor_role'] == 1:
                actor['location'] = location
                actor['rotation_degrees'] = rotation_degrees

        return scenario_data

    def run(
            self,
            browser,
            host=HOST,
            lobby=LOBBY,
            static_instructions=False,
            deadline=None
    ):
        browser.refresh()
        self.display.show_cross()
        t0 = time.time()

        logger.info(f"Trying to connect to {host} and lobby {lobby}")
        client = RemoteClient(url=host, render=False, lobby_name=lobby)
        connected, reason = client.Connect()
        logger.info(f"Connected: {connected}")

        logger.info(f"Attaching scenario.")
        game = None
        while game is None:
            time.sleep(0.1)
            game, reason = client.AttachToScenario(
                scenario_id='', timeout=timedelta(minutes=1)
            )
        logger.info(f"Attached scenario.")

        logger.info('Scenario path: %s' % self.scenario_path)

        scenario_data = self.scenario_data

        objectives = scenario_data['objectives']
        if static_instructions:
            instruction = ''
            while objectives:
                _instruction = objectives.pop(0)['text']
                if instruction:
                    instruction += ' Then ' + _instruction[0].lower() + _instruction[1:]
                else:
                    instruction += _instruction
            objective = OBJECTIVE.copy()
            objective['text'] = instruction
            objectives = [objective]
        scenario_data['objectives'] = objectives

        # Store initial instructions for tracking
        self.all_instructions = [obj['text'] for obj in objectives]
        if self.all_instructions:
            self.current_instruction = self.all_instructions[0]

        if self.state is not None:
            scenario_data = self.update_from_state(scenario_data)
        n_cards_prev = len(scenario_data['prop_update']['props'])

        scenario_data_json = json.dumps(scenario_data)

        self.reset()
        now = datetime.utcnow()
        scenario_data['turn_state']['game_start'] = now
        scenario_data['turn_state']['turn_end'] = now + timedelta(seconds=3600)
        game.step(Action.LoadScenario(scenario_data_json))
        logger.info(f"Loaded...")

        # The following JavaScript injection maps the BUTTONBOX keys to movement before they reach Unity
        # Change the numberical mappings below to remap the buttonboxes
        browser_key_mapping = """
            const keyMapping = {
                '1': 'KeyS',      // Right thumb -> S (select)
                '2': 'ArrowUp',   // Right index -> Up
                '3': 'ArrowDown', // Right middle -> Down  
                '7': 'ArrowRight',// Left index -> Right
                '8': 'ArrowLeft',  // Left middle -> Left
                ' ': 'KeyS'       // Spacebar remapping to select
            };

            function createKeyEvent(type, keyCode, key) {
                return new KeyboardEvent(type, {
                    key: key,
                    code: keyCode,
                    keyCode: keyCode === 'KeyS' ? 83 : (
                        keyCode === 'ArrowUp' ? 38 :
                        keyCode === 'ArrowDown' ? 40 :
                        keyCode === 'ArrowRight' ? 39 :
                        keyCode === 'ArrowLeft' ? 37 : 0
                    ),
                    which: keyCode === 'KeyS' ? 83 : (
                        keyCode === 'ArrowUp' ? 38 :
                        keyCode === 'ArrowDown' ? 40 :
                        keyCode === 'ArrowRight' ? 39 :
                        keyCode === 'ArrowLeft' ? 37 : 0
                    ),
                    bubbles: true,
                    cancelable: true
                });
            }

            // Intercept and convert numerical keys to arrow keys
            document.addEventListener('keydown', function(event) {
                if (keyMapping[event.key]) {
                    // Prevent the original numerical key from reaching Unity
                    event.preventDefault();
                    event.stopPropagation();

                    // Create and dispatch the mapped key event
                    const mappedKey = keyMapping[event.key];
                    const newEvent = createKeyEvent('keydown', mappedKey, 
                        mappedKey.startsWith('Arrow') ? mappedKey : 's');

                    // Target the Unity canvas specifically
                    const unityCanvas = document.querySelector('canvas');
                    if (unityCanvas) {
                        unityCanvas.dispatchEvent(newEvent);
                    } else {
                        document.dispatchEvent(newEvent);
                    }

                    console.log(`Mapped ${event.key} to ${mappedKey}`);
                    return false;
                }
            }, true); // Use capture phase

            document.addEventListener('keyup', function(event) {
                if (keyMapping[event.key]) {
                    event.preventDefault();
                    event.stopPropagation();

                    const mappedKey = keyMapping[event.key];
                    const newEvent = createKeyEvent('keyup', mappedKey,
                        mappedKey.startsWith('Arrow') ? mappedKey : 's');

                    const unityCanvas = document.querySelector('canvas');
                    if (unityCanvas) {
                        unityCanvas.dispatchEvent(newEvent);
                    } else {
                        document.dispatchEvent(newEvent);
                    }

                    return false;
                }
            }, true);
            """
        browser.execute_script(browser_key_mapping)
        logger.info("Injected key mapping script into browser")

        time.sleep(max(0., 3 - (time.time() - t0)))

        self.display.hide()

        self.start_time = time.time()

        prev_cards = None

        while not self.over:
            game_state = game.step(Action.NoopAction())
            (
                map,
                cards,
                turn_state,
                instructions,
                actors,
                live_feedback,
            ) = game_state

            current_cards = list(cards)  # Convert dict_values to list

            # Card selection changes for CSV
            if prev_cards:
                target_ids = scenario_data.get('target_card_ids', [])
                self.detect_card_selections(prev_cards, current_cards, target_ids)

            # Track instruction changes
            if instructions:
                current_instruction_text = instructions[0].text if instructions[0] else "No instruction"
                if current_instruction_text != self.current_instruction:
                    self.current_instruction = current_instruction_text
                    if current_instruction_text not in self.all_instructions:
                        self.all_instructions.append(current_instruction_text)

            n_cards = len(cards)
            if n_cards < n_cards_prev:
                t = time.time()
                self.target_found_times.append(t)
                self.targets_found += 1
                n_cards_prev = n_cards

                # Log this as a correct selection
                self.card_selections.append({
                    'timestamp': t,
                    'type': 'correct',
                    'card_id': f'target_{self.targets_found}',  # Generic ID for correct selections
                    'instruction': self.current_instruction
                })

                print('Pressing d')
                KEYBOARD.press('d')
                KEYBOARD.release('d')

            n_instructions = len([x for x in instructions if not x.completed])
            if not n_instructions:
                self.over = True
                self.success = True
            elif game.over():
                self.over = True
            elif deadline and time.time() > deadline:
                self.over = True
                self.interrupted = True

            self.n_moves += 1

            prev_cards = current_cards

        self.duration = time.time() - self.start_time
        game.instructions = []
        game.queued_messages = []

        return game_state

    def get_performance_summary(self):
        """Get comprehensive performance data for CSV logging"""
        # Count incorrect selections
        incorrect_selections = [sel for sel in self.card_selections if sel['type'] == 'incorrect']
        incorrect_card_ids = [sel['card_properties'] for sel in incorrect_selections]

        # Get the first instruction
        primary_instruction = self.current_instruction if self.current_instruction else "No instruction available"
        if not primary_instruction and self.all_instructions:
            primary_instruction = self.all_instructions[0]

        return {
            'cards_correct': self.targets_found,
            'cards_incorrect': len(incorrect_selections),
            'incorrect_card_ids': incorrect_card_ids,
            'instructions_text': primary_instruction,
            'all_instructions': self.all_instructions,
            'total_card_selections': len(self.card_selections)
        }

    def get_card_description(self, card):
        """Get human-readable description of card properties"""
        try:
            color = card.card_init.color.name if hasattr(card.card_init.color, 'name') else str(card.card_init.color)
            shape = card.card_init.shape.name if hasattr(card.card_init.shape, 'name') else str(card.card_init.shape)
            count = card.card_init.count
            return f"{color}_{shape}_{count}"
        except:
            return f"card_id_{card.id}"

    def detect_card_selections(self, prev_cards, current_cards, target_card_ids):
        """
        Detect card selections by comparing previous and current card states.
        Track both correct and incorrect selections.
        """
        for i in range(min(len(prev_cards), len(current_cards))):
            prev_card = prev_cards[i]
            current_card = current_cards[i]

            # Check if selection state changed from False/0 to True/1
            prev_selected = prev_card.card_init.selected
            curr_selected = current_card.card_init.selected

            if not prev_selected and curr_selected:  # Card was just selected
                self.handle_card_selection(current_card, target_card_ids)

    def handle_card_selection(self, card, target_card_ids):
        """Handle a detected card selection - determine if correct or incorrect"""
        card_description = self.get_card_description(card)

        # Check if this card is a target
        is_correct = card.id in target_card_ids

        selection_record = {
            'timestamp': time.time(),
            'type': 'correct' if is_correct else 'incorrect',
            'card_id': card.id,
            'card_properties': card_description,
            'instruction': self.current_instruction
        }

        self.card_selections.append(selection_record)

        if not is_correct:
            self.incorrect_selections.append(selection_record)
            print(f"Incorrect card selected: {card_description}")
        else:
            print(f"Correct card selected: {card_description}")

    def reset(self):
        self.over = False
        self.success = False
        self.interrupted = False
        self.start_time = None
        self.target_found_times = []
        self.targets_found = 0
        self.n_moves = 0
        self.duration = None

        # Reset enhanced tracking
        self.card_selections = []
        self.incorrect_selections = []
        self.current_instruction = None
        # Note: Don't reset all_instructions as those come from scenario data

    def results(self):
        out = dict(
            success=self.success,
            interrupted=self.interrupted,
            start_time=self.start_time,
            duration=self.duration,
            targets_found=self.targets_found,
            target_found_times=self.target_found_times,
            n_moves=self.n_moves
        )

        # Add performance summary
        out.update(self.get_performance_summary())

        return out


class Run:
    def __init__(
            self,
            scenario_paths,
            n_trials=N_TRIALS,
            subject_kwargs=None,
            display=None
    ):
        self.scenario_paths = scenario_paths
        self.n_trials = n_trials
        self.subject_kwargs = subject_kwargs
        if display is None:
            self.display = Display()
        else:
            self.display = display
        self.subject_kwargs = subject_kwargs
        self.start_time = None
        self.behavioral = None

    def run(
            self,
            browser,
            host=HOST,
            lobby=LOBBY,
            static_instructions=False,
            run_duration=MAX_RUN_DURATION
    ):
        self.display.await_trigger()
        self.start_time = time.time()

        behavioral = []
        if run_duration:
            deadline = time.time() + run_duration
        else:
            deadline = None

        for i, scenario_path in enumerate(self.scenario_paths):
            game_state = None
            for j in range(self.n_trials):
                trial = Trial(
                    scenario_path,
                    state=game_state,
                    subject_kwargs=self.subject_kwargs,
                    display=self.display
                )
                game_state = trial.run(
                    browser,
                    host=host,
                    lobby=lobby,
                    static_instructions=static_instructions,
                    deadline=deadline
                )
                row = trial.results()
                success = row['success']
                interrupted = row['interrupted']
                row['scenario_path'] = scenario_path
                behavioral.append(row)
                time.sleep(0.3)
                # self.display.show_result(success, interrupted)
                # time.sleep(1)
                if interrupted:
                    break
        behavioral = pd.DataFrame(behavioral)
        self.behavioral = behavioral

    def results(self):
        behavioral = self.behavioral.copy()
        behavioral.start_time -= self.start_time
        return behavioral


class Display:
    def __init__(self, font_size=FONT_SIZE):
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.W, self.H = screen.get_size()
        self.font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        self.screen = pygame.display.set_mode((0, 0), pygame.HIDDEN)
        self.visible = False

    def hide(self):
        if self.visible:
            self.visible = False
            pygame.display.set_mode((0, 0), pygame.HIDDEN)
            pygame.display.flip()

    def show(self):
        if not self.visible:
            self.visible = True
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def clear(self):
        assert self.visible, 'Cannot clear display when it is hidden'
        self.screen.fill(pygame.Color('white'))
        pygame.display.flip()

    def draw_text(self, text, color="black"):
        assert self.visible, 'Cannot draw text to display when it is hidden'
        text = self.font.render(text, True, pygame.Color(color))
        text_rect = text.get_rect(center=(self.W / 2, self.H / 2))
        self.screen.blit(text, text_rect)

    def show_cross(self, size=50):
        self.show()
        self.clear()

        x = self.W // 2
        y = self.H // 2
        length = size
        width = size // 10
        pygame.draw.line(self.screen, 'black', (x, y - length), (x, y + length), width)
        pygame.draw.line(self.screen, 'black', (x - length, y), (x + length, y), width)
        pygame.display.flip()

    def await_keypress(
            self,
            keycode,
            instruction
    ):
        self.show()
        self.clear()

        success = False
        escape = False

        self.draw_text(instruction)
        pygame.display.flip()
        while not (success or escape):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and (keycode is None or event.key == keycode):
                    success = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    escape = True
            pygame.time.wait(10)

        return success

    def await_trigger(self, in_scanner=True):
        if in_scanner:
            instruction = "Waiting for scanner..."
            key = pygame.K_EQUALS
        else:
            instruction = 'Press any key to continue...'
            key = None

        self.await_keypress(key, instruction)

    def show_result(self, success, interrupted):
        self.show()
        self.clear()

        if success:
            self.draw_text('You won!', color='blue')
        elif interrupted:
            self.draw_text('Time limit reached')
        else:
            self.draw_text('You lost', color='red')
        pygame.display.flip()

    def show_complete(self):
        self.show()
        self.clear()

        self.draw_text('Run complete.')
        pygame.display.flip()


def grab_pygame_focus_linux():
    """
    Takes control from the Unity webclient game window and passes to pygame for rating input.
    This method uses linux-specific packages, so it will work smoothly on the scanning laptop
    but may need to be adapted for use on Windows / Mac
    """
    try:
        import subprocess

        # Method 1: Try wmctrl if available
        try:
            # Get the pygame window ID
            result = subprocess.run(['xdotool', 'search', '--name', 'pygame'],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]

                # Focus the window
                subprocess.run(['xdotool', 'windowactivate', window_id],
                               check=False, timeout=1)

                # Raise the window to front
                subprocess.run(['xdotool', 'windowraise', window_id],
                               check=False, timeout=1)

                logger.info("Successfully focused pygame window using xdotool")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Method 2: Try wmctrl
        try:
            subprocess.run(['wmctrl', '-a', 'pygame'],
                           check=False, capture_output=True, timeout=1)
            logger.info("Attempted to focus pygame window using wmctrl")
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Method 3: Try xwininfo and xdotool combination
        try:
            # Find pygame window
            result = subprocess.run(['xwininfo', '-name', 'pygame'],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Extract window ID from xwininfo output
                for line in result.stdout.split('\n'):
                    if 'Window id:' in line:
                        window_id = line.split()[3]
                        subprocess.run(['xdotool', 'windowactivate', window_id],
                                       check=False, timeout=1)
                        logger.info("Successfully focused pygame using xwininfo+xdotool")
                        return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass

        logger.warning("Could not focus pygame window - no suitable tools available")
        return False

    except Exception as e:
        logger.warning(f"Error trying to focus pygame window: {e}")
        return False


def prompt_for_rating(display, browser=None):
    """
    Show prompt on display asking user to input difficulty rating (1-7).
    Uses JavaScript to blur Unity game focus without minimizing window
    NOTE: this calls linux-specific methods, so adapt for Windows / ARM if necessary in future
    """
    # first, use JavaScript to blur the Unity game and disable participabt input
    if browser:
        try:
            logger.info("Blurring Unity game and disabling input capture...")

            # remove focus from Unity by sending script to JS (Claude helped me write this part)
            blur_script = """
           // Blur any active element (likely the Unity canvas)
           if (document.activeElement) {
               document.activeElement.blur();
           }

           // Find and blur the Unity canvas specifically
           const unityCanvas = document.querySelector('canvas');
           if (unityCanvas) {
               unityCanvas.blur();
               unityCanvas.style.pointerEvents = 'none';  // Disable mouse events
               unityCanvas.tabIndex = -1;  // Remove from tab order

               // Store original styles for restoration
               unityCanvas._originalPointerEvents = unityCanvas.style.pointerEvents;
               unityCanvas._originalTabIndex = unityCanvas.tabIndex;
           }

           // Blur the entire window
           window.blur();

           // Create a temporary overlay div to capture any stray clicks
           const overlay = document.createElement('div');
           overlay.id = 'pygame-feedback-overlay';
           overlay.style.position = 'fixed';
           overlay.style.top = '0';
           overlay.style.left = '0';
           overlay.style.width = '100%';
           overlay.style.height = '100%';
           overlay.style.backgroundColor = 'rgba(0,0,0,0.1)';
           overlay.style.zIndex = '9999';
           overlay.style.cursor = 'not-allowed';
           overlay.innerHTML = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 24px; text-align: center; background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px;">Use the rating window to provide feedback</div>';
           document.body.appendChild(overlay);

           // Disable all form elements temporarily
           const formElements = document.querySelectorAll('input, button, select, textarea');
           formElements.forEach(el => {
               el.disabled = true;
               el._wasDisabled = true;
           });

           // Return success indicator
           return true;
           """

            result = browser.execute_script(blur_script)
            time.sleep(0.5)  # Give time for focus change
            logger.info("Unity game focus successfully removed")

        except Exception as e:
            logger.warning(f"Could not blur Unity game: {e}")

    # set up pygame window
    pygame.event.clear()
    logger.info("Setting up pygame window with focus...")

    display.show()
    display.clear()

    # try to grab focus for pygame window using Linux methods
    focus_grabbed = grab_pygame_focus_linux()

    # show rating prompt
    base_instruction = "Rate the last room's difficulty (1-7):"
    if focus_grabbed:
        focus_instruction = "Press 1-7 to rate"
    else:
        focus_instruction = "Click HERE first, then press 1-7"

    main_text = display.font.render(base_instruction, True, pygame.Color("black"))
    main_rect = main_text.get_rect(center=(display.W / 2, display.H / 2 - 50))
    display.screen.blit(main_text, main_rect)

    focus_color = "blue" if focus_grabbed else "red"
    focus_text = display.font.render(focus_instruction, True, pygame.Color(focus_color))
    focus_rect = focus_text.get_rect(center=(display.W / 2, display.H / 2 + 50))
    display.screen.blit(focus_text, focus_rect)

    pygame.display.flip()

    # collect rating with validation on keypresses
    rating = None
    focus_received = focus_grabbed  # if we successfully grabbed focus, skip manual window click requirement

    valid_keys_mapping = {
        pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
        pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
        # numpad keys (in case laptop has that instead of top row numbers)
        pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4,
        pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7
    }

    def update_display(instruction, input_text="", feedback_color="black"):
        """Update the display with current instruction and input"""
        display.clear()

        # Show main instruction
        main_text = display.font.render(instruction, True, pygame.Color("black"))
        main_rect = main_text.get_rect(center=(display.W / 2, display.H / 2 - 50))
        display.screen.blit(main_text, main_rect)

        # Show current input if any
        if input_text:
            input_display = display.font.render(f"You selected: {input_text}", True, pygame.Color(feedback_color))
            input_rect = input_display.get_rect(center=(display.W / 2, display.H / 2 + 50))
            display.screen.blit(input_display, input_rect)

        pygame.display.flip()

    # Add timeout mechanism (30 seconds)
    start_time = time.time()
    timeout_seconds = 30
    logger.info(f"Starting rating input loop with {timeout_seconds}s timeout")

    while rating is None:
        # Check for timeout
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed > timeout_seconds:
            logger.warning(f"Rating prompt timed out after {timeout_seconds} seconds, using default rating of 4")
            rating = 4
            break

        # Log every 10 seconds to show we're waiting
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            remaining = timeout_seconds - elapsed
            logger.info(f"Still waiting for rating input... {remaining:.0f}s remaining")

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                # User clicked pygame window - this should give it focus
                focus_received = True
                logger.info("Mouse click detected - pygame window should now have focus")

                # Try to grab focus again after click
                grab_pygame_focus_linux()

                update_display(base_instruction, "Window focused - now press 1-7", "blue")

            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                logger.info(f"Key pressed during rating: {event.key} ({key_name})")

                if not focus_received:
                    # First keypress - show that we received input
                    focus_received = True
                    logger.info("First keypress detected - pygame has focus")

                if event.key in valid_keys_mapping:
                    rating = valid_keys_mapping[event.key]
                    logger.info(f"Valid rating received: {rating}")

                    # Show confirmation feedback
                    update_display(base_instruction, f"{rating} (Confirmed!)", "green")
                    pygame.time.wait(1000)  # Show confirmation for 1 second

                elif event.key == pygame.K_ESCAPE:
                    logger.info("Escape key pressed, returning default rating")
                    update_display(base_instruction, "Escaped - using default", "orange")
                    pygame.time.wait(500)
                    rating = 4  # Default rating

                else:
                    # Show invalid key feedback
                    invalid_feedback = f"{key_name} (Invalid - use 1-7)"
                    update_display(base_instruction, invalid_feedback, "red")
                    logger.info(f"Invalid key during rating: {event.key} ({key_name})")

            elif event.type == pygame.QUIT:
                logger.info("Pygame QUIT event received during rating")
                rating = 4  # Default rating
                break

        pygame.time.wait(10)

    # Step 5: Restore Unity game focus and input
    if browser:
        try:
            logger.info("Restoring Unity game focus...")

            restore_script = """
           // Remove the overlay
           const overlay = document.getElementById('pygame-feedback-overlay');
           if (overlay) {
               overlay.remove();
           }

           // Re-enable Unity canvas
           const unityCanvas = document.querySelector('canvas');
           if (unityCanvas) {
               unityCanvas.style.pointerEvents = 'auto';
               unityCanvas.tabIndex = 0;
               unityCanvas.focus();

               // Clean up stored properties
               delete unityCanvas._originalPointerEvents;
               delete unityCanvas._originalTabIndex;
           }

           // Re-enable form elements
           const formElements = document.querySelectorAll('input, button, select, textarea');
           formElements.forEach(el => {
               if (el._wasDisabled) {
                   el.disabled = false;
                   delete el._wasDisabled;
               }
           });

           // Focus the window
           window.focus();

           // Click the canvas to ensure Unity gets focus
           if (unityCanvas) {
               const rect = unityCanvas.getBoundingClientRect();
               const clickEvent = new MouseEvent('click', {
                   clientX: rect.left + rect.width / 2,
                   clientY: rect.top + rect.height / 2,
                   bubbles: true
               });
               unityCanvas.dispatchEvent(clickEvent);
           }

           return true;
           """

            browser.execute_script(restore_script)
            time.sleep(0.3)
            logger.info("Unity game focus restored")

        except Exception as e:
            logger.warning(f"Could not restore Unity focus: {e}")

    return rating if rating is not None else 4


def restore_unity_focus_after_rating(browser):
    """
    Helper function to ensure Unity game has focus after rating collection.
    Call this before starting the next scenario to get back to the Unity webclient window.
    """
    if browser:
        try:
            focus_script = """
           const unityCanvas = document.querySelector('canvas');
           if (unityCanvas) {
               unityCanvas.focus();

               // Simulate a click to ensure Unity gets input focus
               const rect = unityCanvas.getBoundingClientRect();
               const clickEvent = new MouseEvent('click', {
                   clientX: rect.left + rect.width / 2,
                   clientY: rect.top + rect.height / 2,
                   bubbles: true
               });
               unityCanvas.dispatchEvent(clickEvent);

               // Also simulate a key event to "wake up" Unity input system
               const keyEvent = new KeyboardEvent('keydown', {
                   key: ' ',
                   code: 'Space',
                   bubbles: true
               });
               unityCanvas.dispatchEvent(keyEvent);

               const keyUpEvent = new KeyboardEvent('keyup', {
                   key: ' ',
                   code: 'Space',
                   bubbles: true
               });
               unityCanvas.dispatchEvent(keyUpEvent);
           }
           window.focus();
           """
            browser.execute_script(focus_script)
            time.sleep(0.2)
            logger.info("Unity focus restored for next scenario")
        except Exception as e:
            logger.warning(f"Could not restore Unity focus: {e}")


def practice_arrow_keys(display=None):
    """
    Practice function for MRI buttonbox controls.
    Tests up (2), back (3), right (7), left (8), and select (1).
    """
    if display is None:
        display = Display()

    logger.info("Starting buttonbox practice")

    controls = [
        {"name": "UP", "key": "2", "pygame_key": pygame.K_2},
        {"name": "BACK", "key": "3", "pygame_key": pygame.K_3},
        {"name": "RIGHT", "key": "7", "pygame_key": pygame.K_7},
        {"name": "LEFT", "key": "8", "pygame_key": pygame.K_8},
        {"name": "SELECT", "key": "1", "pygame_key": pygame.K_1}  # Using key 1 for select (right thumb)
    ]

    def show_instruction(text, color="black", wait_for_key=False):
        display.clear()
        text_surface = display.font.render(text, True, pygame.Color(color))
        text_rect = text_surface.get_rect(center=(display.W / 2, display.H / 2))
        display.screen.blit(text_surface, text_rect)
        pygame.display.flip()

        if wait_for_key:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        return
                pygame.time.wait(10)

    def load_button_image(button_key):
        """Load button image with fallback for different file extensions and hand positions."""
        import os

        # Try different combinations of hand (RH/LH) and file extension (.jpg/.JPG)
        base_path = os.path.expanduser("~/cb2/src/cb2game/fmri/button_images")
        hands = ["RH", "LH"]

        for hand in hands:
            filename = f"BB-{hand}-{button_key}.jpg"
            filepath = os.path.join(base_path, filename)

            if os.path.exists(filepath):
                try:
                    image = pygame.image.load(filepath)
                    logger.info(f"Loaded button image: {filename}")
                    return image
                except pygame.error as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    continue

        logger.warning(f"No button image found for key {button_key}")
        return None

    # Show intro
    display.show()
    show_instruction("BUTTONBOX PRACTICE - Press any button to start", wait_for_key=True)

    completed = []

    # Test each control - stay on each until successful
    for i, control in enumerate(controls):
        logger.info(f"Testing {control['name']} control")

        button_completed = False

        while not button_completed:
            # Show prompt with progress and button info
            display.clear()
            display.screen.fill(pygame.Color("white"))  # white background for instructions
            pygame.display.flip()

            # Progress
            progress = f"Test {i + 1} of {len(controls)}"
            progress_surface = display.font.render(progress, True, pygame.Color("gray"))
            display.screen.blit(progress_surface, (display.W // 2 - progress_surface.get_width() // 2, 80))

            # Main instruction
            instruction = f"Press the {control['name']} button"
            instruction_surface = display.font.render(instruction, True, pygame.Color("black"))
            display.screen.blit(instruction_surface,
                                (display.W // 2 - instruction_surface.get_width() // 2, display.H // 2 - 250))

            # Load and display button image
            button_image = load_button_image(control['key'])
            if button_image:
                # Scale image to a reasonable size (max 200px on either dimension while maintaining aspect ratio)
                original_size = button_image.get_size()
                max_size = 600

                if original_size[0] > original_size[1]:  # Wider than tall
                    new_width = min(max_size, original_size[0])
                    new_height = int(original_size[1] * (new_width / original_size[0]))
                else:  # Taller than wide or square
                    new_height = min(max_size, original_size[1])
                    new_width = int(original_size[0] * (new_height / original_size[1]))

                scaled_image = pygame.transform.scale(button_image, (new_width, new_height))
                image_rect = (display.W // 2 - new_width // 2, display.H // 2 - 50)
                display.screen.blit(scaled_image, image_rect)
            else:
                # Fallback placeholder if image not found
                placeholder = f"[{control['name']} BUTTON IMAGE]"
                placeholder_surface = pygame.font.Font(pygame.font.get_default_font(), 40).render(placeholder, True,
                                                                                                  pygame.Color(
                                                                                                      "orange"))
                display.screen.blit(placeholder_surface,
                                    (display.W // 2 - placeholder_surface.get_width() // 2, display.H // 2 + 100))

            pygame.display.flip()

            # Wait for correct button (no timeout - stay until correct)
            waiting_for_correct = True
            while waiting_for_correct:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key_name = pygame.key.name(event.key)
                        logger.info(f"Key pressed during {control['name']} test: {event.key} ({key_name})")

                        if event.key == control['pygame_key']:
                            # Correct button
                            completed.append(control['name'])
                            button_completed = True
                            waiting_for_correct = False
                            logger.info(f"Correct {control['name']} button pressed")

                            show_instruction(f"{control['name']} works!", "green")
                            pygame.time.wait(3000)
                            break
                        else:
                            # Wrong button - show error and continue waiting
                            show_instruction(f"Wrong button!", "red")
                            pygame.time.wait(1000)
                            waiting_for_correct = False  # Exit inner loop to redraw prompt
                            break
                    elif event.type == pygame.QUIT:
                        return False

                if waiting_for_correct:
                    pygame.time.wait(10)

    # Show completion
    show_instruction("Button practice complete!", "green", wait_for_key=True)
    logger.info("Buttonbox practice completed successfully - all buttons tested")

    return True


def run_palindrome_behavioral(
        subject_id,
        run_number,
        run_set,
        task_difficulty,
        linguistic_complexity,
        browser,
        materials_dir=MATERIALS_DIR,
        display=None,
        host=HOST,
        lobby=LOBBY,
        no_test_button_box=False,
):
    import random
    if display is None:
        display = Display()

    subject_kwargs = {"subject_id": subject_id, "run": run_number}

    # Define the 4 scenario condition tuples for the palindrome pattern
    conditions = [
        (task_difficulty, linguistic_complexity),
        (0, linguistic_complexity),
        (task_difficulty, 0),
        (0, 0)
    ]

    # Validate required scenario files
    logger.info(f"Validating scenario files for {run_set}...")
    files_exist, missing_files, available_files = validate_scenario_files(
        materials_dir, run_set, conditions
    )

    if not files_exist:
        logger.error("Missing required scenario files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        raise FileNotFoundError(f"Cannot run experiment - missing scenario files: {missing_files}")

    # Test buttonbox if needed
    if not no_test_button_box:
        logger.info("Testing buttonbox controls before starting behavioral experiment...")
        practice_success = practice_arrow_keys(display)
        if not practice_success:
            logger.warning("Buttonbox practice was not fully successful, but continuing with experiment")

    # Randomize the initial 4 scenario orders
    randomized_conditions = random.sample(conditions, k=4)

    # Load scenario paths for each condition in randomized order
    scenario_paths = []
    condition_sequence = []  # Track the condition sequence for logging

    for (env, ling) in randomized_conditions:
        # Get all scenarios for this condition
        condition_scenarios = available_files[(env, ling)]
        # Randomly select just ONE scenario from this condition
        selected_scenario = random.choice(condition_scenarios)
        scenario_paths.append(selected_scenario['full_path'])
        condition_sequence.append((env, ling))
        logger.info(f"Selected scenario for condition t{env}_l{ling}: {selected_scenario['filename']}")

    # Create full palindrome: forward + reverse
    palindrome_paths = scenario_paths + list(reversed(scenario_paths))
    palindrome_conditions = condition_sequence + list(reversed(condition_sequence))

    logger.info(f"Running palindrome behavioral with {len(palindrome_paths)} scenarios")
    logger.info(f"Condition order: {randomized_conditions} forward + reversed")

    # Record experiment start time
    experiment_start_utc = datetime.utcnow().isoformat() + "Z"

    # Run each scenario
    for i, (scenario_path, condition) in enumerate(zip(palindrome_paths, palindrome_conditions)):
        logger.info(f"Running scenario {i + 1}/{len(palindrome_paths)}: {os.path.basename(scenario_path)}")

        # Determine palindrome half
        palindrome_half = "forward" if i < 4 else "reverse"

        # Ensure Unity has focus before starting the trial
        restore_unity_focus_after_rating(browser)

        trial = Trial(
            scenario_path,
            subject_kwargs=subject_kwargs,
            display=display
        )

        # Run trial with time limit
        start_trial_time = time.time()
        game_state = trial.run(
            browser,
            host=host,
            lobby=lobby,
            static_instructions=False,
            deadline=time.time() + 60  # 60 seconds per scenario
        )

        trial_end_time = time.time()
        logger.info(f"Trial {i + 1} completed in {trial_end_time - start_trial_time:.2f} seconds")

        # Clear pygame events before rating prompt
        pygame.event.clear()
        time.sleep(0.5)

        # Get difficulty rating
        logger.info(f"Showing rating prompt for trial {i + 1}")
        rating = prompt_for_rating(display, browser)
        logger.info(f"Rating received for trial {i + 1}: {rating}")

        # Extract performance data
        performance_data = extract_game_performance(game_state, trial)

        # Prepare trial data for CSV
        task_diff, ling_comp = condition
        trial_data = {
            'subject_id': subject_id,
            'run_number': run_number,
            'experiment_start_utc': experiment_start_utc,
            'scenario_position': i + 1,
            'condition_code': condition_to_code(task_diff, ling_comp),
            'condition_detail': condition_to_detail(task_diff, ling_comp),
            'palindrome_half': palindrome_half,
            'scenario_filename': os.path.basename(scenario_path),
            'instructions_text': performance_data['instructions_text'],
            'cards_correct': performance_data['cards_correct'],
            'cards_incorrect': performance_data['cards_incorrect'],
            'incorrect_card_ids': json.dumps(performance_data['incorrect_card_ids']),
            'difficulty_rating': rating,
            'trial_duration_seconds': trial.duration,
            'trial_completed_utc': datetime.utcnow().isoformat() + "Z"
        }

        # Write to CSV
        write_trial_data_to_csv(trial_data)

        logger.info(f"Trial {i + 1} data logged. Pausing before next trial...")
        time.sleep(1)  # Pause between trials

    logger.info(f"Behavioral run complete. Data logged to {CSV_FILE_PATH}")

    # Show completion screen
    display.show_complete()
    time.sleep(2)


def normal_main(
        subject_id,
        run_number,
        run_set,
        task_difficulty,
        linguistic_complexity,
        browser,
        materials_dir=MATERIALS_DIR,
        n_trials=None,
        host=HOST,
        lobby=LOBBY,
        static_instructions=False,
        no_test_button_box=False,
        not_in_scanner=False,
        display=None
):
    if display is None:
        display = Display()

    subject_kwargs = {"subject_id": subject_id, "run": run_number}
    logger.info(f'SUBJECT_ID: {subject_kwargs["subject_id"]}\nRUN#: {repr(subject_kwargs["run"])}')

    # ask the participant to test their controls - need to implement this for in the scanner
    if not no_test_button_box:
        practice_success = practice_arrow_keys(display)  # ADD display parameter
        if not practice_success:
            logger.warning("Buttonbox practice was not fully successful, but continuing with experiment")

    # validate that thr scenario files exist for the specified condition
    conditions = [(task_difficulty, linguistic_complexity)]
    files_exist, missing_files, available_files = validate_scenario_files(
        materials_dir, run_set, conditions
    )

    if not files_exist:
        logger.error("Missing required scenario files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        raise FileNotFoundError(f"Cannot run experiment - missing scenario files: {missing_files}")

    # Build scenario paths using the validated files
    condition = (task_difficulty, linguistic_complexity)
    scenario_data = available_files[condition]
    scenario_paths = [scenario['full_path'] for scenario in scenario_data]
    scenario_paths = list(np.random.permutation(scenario_paths))

    if n_trials:
        scenario_paths = scenario_paths[:n_trials]

    logger.info(run_set[0].upper() + run_set[1:])
    logger.info(f'Condition: T{task_difficulty} L{linguistic_complexity}')
    logger.info(f'Found {len(scenario_paths)} scenarios for this condition')

    run = Run(
        scenario_paths=scenario_paths,
        subject_kwargs=subject_kwargs,
        display=display
    )
    run.run(
        browser,
        host=host,
        lobby=lobby,
        static_instructions=static_instructions
    )
    behavioral = run.results()
    behavioral['task_difficulty'] = task_difficulty
    behavioral['linguistic_complexity'] = linguistic_complexity
    if not os.path.exists('behavioral'):
        os.makedirs('behavioral')
    behavioral.to_csv(os.path.join('behavioral', f'{subject_id}_run{run_number}.csv'), index=False)

    return None  # Fixed: was returning undefined 'client'


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fmri")
    parser.add_argument('subject_id')
    parser.add_argument('run_number')
    parser.add_argument('run_set')
    parser.add_argument('task_difficulty', type=int)
    parser.add_argument('linguistic_complexity', type=int)
    parser.add_argument('--materials-dir', default=MATERIALS_DIR)
    parser.add_argument('--n-trials', default=None)
    parser.add_argument("--static-instructions", action="store_true")
    parser.add_argument("--no-test-button-box", action="store_true")
    parser.add_argument("--not-in-scanner", action="store_true")
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--lobby", type=str, default=LOBBY)

    # Behavioral experiment flag for randomized palindrome selection and user feedback
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Run palindrome behavioral protocol with difficulty ratings after each scenario"
    )

    args = parser.parse_args()

    subject_id = args.subject_id
    run_number = int(args.run_number)
    run_set = args.run_set
    if not run_set.startswith('runset'):
        run_set = f'runset_{run_set}'
    task_difficulty = int(args.task_difficulty)
    linguistic_complexity = int(args.linguistic_complexity > 0)
    materials_dir = args.materials_dir
    n_trials = args.n_trials
    if n_trials is not None:
        n_trials = int(n_trials)
    static_instructions = args.static_instructions
    no_test_button_box = args.no_test_button_box
    not_in_scanner = args.not_in_scanner
    host = args.host
    lobby = args.lobby

    browser = open_browser(fullscreen=True)
    url = f"{host}/play?lobby_name={lobby}&auto=join_game_queue"
    browser.get(url)
    display = Display()

    excp = None
    client = None

    try:
        if args.behavioral:
            run_palindrome_behavioral(
                subject_id=subject_id,
                run_number=run_number,
                run_set=run_set,
                task_difficulty=task_difficulty,
                linguistic_complexity=linguistic_complexity,
                browser=browser,
                materials_dir=materials_dir,
                display=display,
                host=host,
                lobby=lobby,
                no_test_button_box=no_test_button_box,
            )
        else:
            client = normal_main(
                subject_id=subject_id,
                run_number=run_number,
                run_set=run_set,
                task_difficulty=task_difficulty,
                linguistic_complexity=linguistic_complexity,
                browser=browser,
                materials_dir=materials_dir,
                n_trials=n_trials,
                host=host,
                lobby=lobby,
                static_instructions=static_instructions,
                no_test_button_box=no_test_button_box,
                not_in_scanner=not_in_scanner,
                display=display,
            )
    except Exception as e:
        excp = e

    display.show_complete()
    browser.close()
    if client is not None:
        client.Reset()
    logger.info("Run complete.")
    if excp is not None:
        raise excp
    time.sleep(2)