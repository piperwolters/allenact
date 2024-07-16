import os
import sys
sys.path.append('/home/piperw/digirl/')

from digirl.environment.android.autoui_utils import autoui_prepare_prompt, autoui_translate_action, ImageFeatureExtractor
from digirl.environment.android.env import AndroidEmulator
from digirl.environment.android.evaluate import EndResultEvaluator

# Arguments
cache_avd_name = 'test1'
max_steps = 10
emulator_path = '/home/piperw/.android/emulator/emulator'
appium_server_url = 'http://localhost:6652'
run_headless = True
udid = 'emulator-5554'
feature_extractor = ImageFeatureExtractor('cpu')
evaluator = EndResultEvaluator(gemini_key=api_key, task_set="general")
tmp_path = '/home/piperw/logs/digirl-general-online/images/test1'
prepare_prompt = autoui_prepare_prompt
translate_action = autoui_translate_action
save_images = True
task_id = 0
task_split = 'train'
sample_mode = 'random'
record= False

all_tasks = [
    'Check the settings for the Pandora app',
    "What's a good restaurant in Los Angeles?",
    'How old is the earth?',
    "What's the price of the LG TV?",
    'Play the new Katy Perry video on YouTube',
    'How much does a 2 bedroom apartment rent for in San Francisco?',
    'What is the capital of Norway?',
    'Google the capital of Canada',
    'What is the speed of a bicycle?'
]

random_actions = [
    f'Action Decision: "action_type": "DUAL_POINT", "touch_point": "[0.7853, 0.6861]", "lift_point": "[0.7853, 0.6861]", "typed_text": ""',
    f'Action Decision: "action_type": "DUAL_POINT", "touch_point": "[0.8, 0.5]", "lift_point": "[0.2, 0.5]", "typed_text": "" ',
    f'Action Decision: "action_type": "DUAL_POINT", "touch_point": "[0.3531, 0.5111]", "lift_point": "[0.3531, 0.5111]", "typed_text": "" ',
    f'Action Decision: "action_type": "DUAL_POINT", "touch_point": "[0.7853, 0.6861]", "lift_point": "[0.7853, 0.6861]", "typed_text": ""',
    f'Action Decision: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "pandora"',
    f'Action Decision: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""',
    f'Action Decision: "action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""',
    f'Action Decision: "action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
]

def get_env():
    print("Initializing Android Emulator...")
    return AndroidEmulator(
                avd_name=cache_avd_name,
                max_steps=max_steps,
                emulator_path=emulator_path,
                appium_server_url=appium_server_url,
                no_window=run_headless,
                udid = udid,
                feature_extractor = feature_extractor,
                prepare_prompt = prepare_prompt,
                translate_action = translate_action,
                all_tasks = all_tasks,
                evaluator = evaluator,
                temp_path = os.path.join(tmp_path, cache_avd_name),
                save_images = save_images,
                task_id=task_id,
                task_split=task_split,
                sample_mode=sample_mode,
                record=record
        )
