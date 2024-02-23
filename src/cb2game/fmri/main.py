import logging
import json
import os
import argparse
import pygame
import time
from datetime import datetime, timedelta
from pynput import keyboard
import numpy as np
import pandas as pd

from cb2game.pyclient.remote_client import RemoteClient
from cb2game.pyclient.game_endpoint import Action
from cb2game.fmri.utils import OBJECTIVE, MAX_TRIAL_DURATION, MAX_RUN_DURATION, open_browser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
FONT_SIZE = 75
HOST = "http://localhost:8080"
LOBBY = "scenario-lobby"
MATERIALS_DIR = 'materials'
MAX_TARGETS = None
KEYBOARD = keyboard.Controller()


class Trial:
    BUTTONBOX_MAP = {
        '1': keyboard.Key.up,
        '2': 's',
        '3': keyboard.Key.left,
        '4': keyboard.Key.right,
        'g': keyboard.Key.up,
        'f': 's',
        'l': keyboard.Key.left,
        ';': keyboard.Key.right,
    }

    def __init__(
            self,
            scenario_path,
            subject_kwargs=None,
            display=None,
            max_targets=MAX_TARGETS,
            max_trial_duration=MAX_TRIAL_DURATION
    ):
        logger.info(f"Loading scenario...")
        self.scenario_path = scenario_path
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
        self.target_start_times = {}
        self.target_found_times = {}
        self.targets_found = 0
        self.n_moves = 0
        self.duration = None

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

    def on_buttonbox_press(self, key):
        try:
            char = key.char
            if char in Trial.BUTTONBOX_MAP:
                KEYBOARD.press(Trial.BUTTONBOX_MAP[char])

        except AttributeError:
            pass

    def on_buttonbox_release(self, key):
        try:
            char = key.char
            if char in Trial.BUTTONBOX_MAP:
                KEYBOARD.release(Trial.BUTTONBOX_MAP[char])
        except AttributeError:
            pass

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

        scenario_data_json = json.dumps(scenario_data)

        self.reset()
        now = datetime.utcnow()
        scenario_data['turn_state']['game_start'] = now
        scenario_data['turn_state']['turn_end'] = now + timedelta(seconds=3600)
        game.step(Action.LoadScenario(scenario_data_json))
        target_card_ids = scenario_data.get('target_card_ids', [])
        target_card_ids = [x[0] for x in target_card_ids]
        target_card_id = target_card_ids.pop(0)
        target_card_id_set = set(target_card_ids)
        target_card_ix = 1
        logger.info(f"Loaded...")

        listener = keyboard.Listener(
            on_press=self.on_buttonbox_press,
            on_release=self.on_buttonbox_release)
        listener.start()

        time.sleep(max(0., 3 - (time.time() - t0)))

        self.display.hide()

        self.start_time = time.time()

        while not self.over:
            (
                map,
                cards,
                turn_state,
                instructions,
                actors,
                live_feedback,
            ) = game.step(Action.NoopAction())

            cards_selected = set()
            cards_unselected = set()
            for card in cards:
                if card.card_init.selected:
                    cards_selected.add(card.id)
                else:
                    cards_unselected.add(card.id)
            if (not (cards_unselected & target_card_id_set) and   # No target cards are unselected
                    not (cards_selected - target_card_id_set)):   # No non-target cards are selected
                self.over = True
                self.success = True
                KEYBOARD.press('d')
                KEYBOARD.release('d')
            elif game.over():
                self.over = True
                self.success = (not (cards_unselected & target_card_id_set) and
                                not (cards_selected - target_card_id_set))
            elif deadline and time.time() > deadline:
                self.over = True
                self.interrupted = True
            elif not static_instructions:
                if target_card_id in cards_selected:
                    t = time.time
                    self.target_found_times['target_%s_found_time' % target_card_ix] = t
                    target_card_ix += 1
                    self.target_start_times['target_%s_start_time' % target_card_ix] = t
                    KEYBOARD.press('d')
                    KEYBOARD.release('d')
                    if len(target_card_ids):
                        target_card_id = target_card_ids.pop(0)
                    else:
                        target_card_id = None
            self.targets_found = len(target_card_id_set & cards_selected)
            self.n_moves += 1

        self.duration = time.time() - self.start_time
        game.instructions = []
        game.queued_messages = []
        listener.stop()

    def reset(self):
        self.over = False
        self.success = False
        self.interrupted = False
        self.start_time = None
        self.target_start_times = {}
        self.target_found_times = {}
        self.targets_found = 0
        self.n_moves = 0
        self.duration = None

    def results(self):
        out = dict(
            success=self.success,
            interrupted=self.interrupted,
            start_time=self.start_time,
            duration=self.duration,
            targets_found=self.targets_found,
            n_moves=self.n_moves
        )
        out.update(self.target_start_times)
        out.update(self.target_found_times)

        return out


class Run:
    def __init__(
            self,
            scenario_paths,
            subject_kwargs=None,
            display=None
    ):
        self.scenario_paths = scenario_paths
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
            deadline=None

        for i, scenario_path in enumerate(self.scenario_paths):
            trial = Trial(scenario_path, subject_kwargs=self.subject_kwargs, display=self.display)
            trial.run(
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
            self.display.show_result(success, interrupted)
            time.sleep(1)
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


def main(
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

    # ask the participant to test their controls
    if not no_test_button_box:
        practice_arrow_keys()

    scenario_paths = []
    scenario_ids = set()
    for path in os.listdir(os.path.join(materials_dir, run_set)):
        if path.startswith('scenario'):
            scenario_id = int(path.split('_')[1])
            scenario_ids.add(scenario_id)
    scenario_ids = sorted(list(scenario_ids))
    for scenario_id in scenario_ids:
        scenario_path = os.path.join(
            materials_dir, run_set, 'scenario_%04d_t%s_l%s.json' % (scenario_id, task_difficulty, linguistic_complexity)
        )
        scenario_paths.append(scenario_path)
    scenario_paths = list(np.random.permutation(scenario_paths))
    if n_trials:
        scenario_paths = scenario_paths[:n_trials]

    logger.info(run_set[0].upper() + run_set[1:])
    logger.info(f'Condition: T{task_difficulty} L{linguistic_complexity}')

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

    return client


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

    args = parser.parse_args()

    subject_id = args.subject_id
    run_number = int(args.run_number)
    run_set = args.run_set
    if not run_set.startswith('runset'):
        run_set = f'runset_{run_set}'
    task_difficulty = int(args.task_difficulty > 0)
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
        client = main(
            subject_id,
            run_number,
            run_set,
            task_difficulty,
            linguistic_complexity,
            browser,
            materials_dir=materials_dir,
            n_trials=n_trials,
            host=host,
            lobby=lobby,
            static_instructions=static_instructions,
            no_test_button_box=no_test_button_box,
            not_in_scanner=not_in_scanner,
            display=display
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
