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
from cb2game.fmri.utils import OBJECTIVE, N_TRIALS, MAX_TRIAL_DURATION, MAX_RUN_DURATION, open_browser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
FONT_SIZE = 75
HOST = "http://localhost:8080"
LOBBY = "scenario-lobby"
MATERIALS_DIR = 'materials'
MAX_TARGETS = None
KEYBOARD = keyboard.Controller()


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
   BUTTONBOX_MAP = {
       '1': 's',  # R thumb
       '2': keyboard.Key.up,  # R index
       '3': keyboard.Key.down,  # R middle
       '7': keyboard.Key.right,  # L index
       '8': keyboard.Key.left,  # L middle

       # Spacebar mapping for selection
       ' ': 's',
   }


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


   def on_buttonbox_press(self, key):
       try:
           char = key.char
           if char in Trial.BUTTONBOX_MAP:
               KEYBOARD.press(Trial.BUTTONBOX_MAP[char])


       except AttributeError:
           if key == keyboard.Key.space:
               KEYBOARD.press('s')  # selection for special keys


   def on_buttonbox_release(self, key):
       try:
           char = key.char
           if char in Trial.BUTTONBOX_MAP:
               KEYBOARD.release(Trial.BUTTONBOX_MAP[char])
       except AttributeError:
           if key == keyboard.Key.space:
               KEYBOARD.release('s')


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


       listener = keyboard.Listener(
           on_press=self.on_buttonbox_press,
           on_release=self.on_buttonbox_release)
       listener.start()


       time.sleep(max(0., 3 - (time.time() - t0)))


       self.display.hide()


       self.start_time = time.time()


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


           n_cards = len(cards)
           if n_cards < n_cards_prev:
               t = time.time
               self.target_found_times.append(t)
               self.targets_found += 1
               n_cards_prev = n_cards
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


       self.duration = time.time() - self.start_time
       game.instructions = []
       game.queued_messages = []
       listener.stop()


       return game_state


   def reset(self):
       self.over = False
       self.success = False
       self.interrupted = False
       self.start_time = None
       self.target_found_times = []
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
           target_found_times=self.target_found_times,
           n_moves=self.n_moves
       )


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




def practice_arrow_keys():
   """Placeholder for the practice function - implement for use with buttonboxes in MRI, incl photos of buttons"""
   logger.info("Practice arrow keys function called")
   pass




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
):
   import random
   if display is None:
       display = Display()


   subject_kwargs = {"subject_id": subject_id, "run": run_number}


   # Define the 4 scenario condition tuples for the palindrome pattern
   # env = environmental condition (task_difficulty)
   # ling = linguistic_complexity
   # Pattern: [env_high ling_high], [env_low ling_high], [env_high ling_low], [env_low ling_low]
   # Use the passed task_difficulty and linguistic_complexity parameters
   conditions = [
       (task_difficulty, linguistic_complexity),
       (0, linguistic_complexity),
       (task_difficulty, 0),
       (0, 0)
   ]

   # validate required scenario files and get set of desired scenarios for palindrome
   logger.info(f"Validating scenario files for {run_set}...")
   files_exist, missing_files, available_files = validate_scenario_files(
       materials_dir, run_set, conditions
   )

   if not files_exist:
       logger.error("Missing required scenario files:")
       for missing in missing_files:
           logger.error(f"  - {missing}")
       raise FileNotFoundError(f"Cannot run experiment - missing scenario files: {missing_files}")

   # Randomize the initial 4 scenario orders
   randomized_conditions = random.sample(conditions, k=4)

   # Load scenario paths for each condition in randomized order
   # Take only ONE scenario per condition for the palindrome pattern
   scenario_paths = []
   for (env, ling) in randomized_conditions:
       # Get all scenarios for this condition
       condition_scenarios = available_files[(env, ling)]

       # Randomly select just ONE scenario from this condition (there are 3 of the same condition per runset)
       selected_scenario = random.choice(condition_scenarios)
       scenario_paths.append(selected_scenario['full_path'])

       logger.info(f"Selected scenario for condition t{env}_l{ling}: {selected_scenario['filename']}")

   # get full palindrome: randomized_conditions forward + reverse
   palindrome_paths = scenario_paths + list(reversed(scenario_paths))

   logger.info(f"Running palindrome behavioral with {len(palindrome_paths)} scenarios")
   logger.info(f"Condition order: {randomized_conditions} forward + reversed")
   logger.info(f"Selected scenarios: {[os.path.basename(p) for p in scenario_paths]}")

   results = []


   # Run each scenario for 60 seconds with prompt for difficulty rating after
   for i, scenario_path in enumerate(palindrome_paths):
       logger.info(f"Running scenario {i + 1}/{len(palindrome_paths)}: {os.path.basename(scenario_path)}")

       # Ensure Unity has focus before starting the trial
       restore_unity_focus_after_rating(browser)

       trial = Trial(
           scenario_path,
           subject_kwargs=subject_kwargs,
           display=display
       )

       # Run trial with 60-second deadline (change to 10s for testing)
       start_trial_time = time.time()
       game_state = trial.run(
           browser,
           host=host,
           lobby=lobby,
           static_instructions=False,
           deadline=time.time() + 60  # 60 seconds per scenario in a block
       )

       trial_end_time = time.time()
       logger.info(f"Trial {i + 1} completed in {trial_end_time - start_trial_time:.2f} seconds")
       logger.info(f"Trial success: {trial.success}, interrupted: {trial.interrupted}")

       # Clear remaining pygame events before showing rating prompt
       pygame.event.clear()
       time.sleep(0.5)

       # prompt participant for difficulty rating 1-7
       logger.info(f"About to show rating prompt for trial {i + 1}")
       rating = prompt_for_rating(display, browser)
       logger.info(f"Rating received for trial {i + 1}: {rating}")

       # store results with timestamps to behavioral_data directory LOCAL to scan laptop
       current_time = datetime.utcnow()
       results.append({
           "scenario": scenario_path,
           "scenario_filename": os.path.basename(scenario_path),
           # "success": trial.success, # this is whether ALL tasks in scenario were completed; unlikely to be useful
           "duration": trial.duration,
           "rating": rating,
           "timestamp_utc": current_time.isoformat() + "Z",
           "timestamp_local": datetime.now().isoformat(),
           "trial_number": i + 1
       })

       logger.info(f"Trial {i + 1} data stored. Pausing before next trial...")
       time.sleep(1)  # pause between trials

   # save ratings to json file with additional metadata - do we want to add anything to this?
   experiment_metadata = {
       "experiment_start_time_utc": datetime.utcnow().isoformat() + "Z",
       "experiment_start_time_local": datetime.now().isoformat(),
       "subject_id": subject_id,
       "run_number": run_number,
       "run_set": run_set,
       "total_scenarios": len(palindrome_paths),
       "condition_order": randomized_conditions,
       "palindrome_pattern": "forward + reversed",
       "trial_duration_seconds": 60  # Update this when changing from test duration
   }

   output_data = {
       "metadata": experiment_metadata,
       "trials": results
   }

   outdir = 'behavioral_ratings'
   if not os.path.exists(outdir):
       os.makedirs(outdir)
   ratings_path = os.path.join(outdir, f"{subject_id}_run{run_number}_ratings.json")

   # absolute path for clearer logging
   abs_ratings_path = os.path.abspath(ratings_path)

   with open(ratings_path, 'w') as f:
       json.dump(output_data, f, indent=2)

   logger.info(f"Behavioral run complete. Ratings saved to:")
   logger.info(f"  Relative path: {ratings_path}")
   logger.info(f"  Absolute path: {abs_ratings_path}")

   # Also log the current working directory for reference
   logger.info(f"  Current working directory: {os.getcwd()}")

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
       practice_arrow_keys()

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