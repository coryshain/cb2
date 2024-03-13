import subprocess
import logging
import numpy as np

logger = logging.getLogger(__file__)

OBJECTIVE = {
  "sender": 2,
  "text": "",
  "uuid": "",
  "completed": False,
  "cancelled": False
}
ASSET_TO_ID=dict(
    PLAYER=0,
    PLAYER_WITH_CAM=1,
    FOLLOWER_BOT=2,
    GROUND_TILE=3,
    GROUND_TILE_ROCKY=4,
    GROUND_TILE_STONES=5,
    GROUND_TILE_STONES_GREENBUSH=6,
    GROUND_TILE_STONES_BROWNBUSH=7,
    GROUND_TILE_STONES_GREYBUSH=8,
    GROUND_TILE_TREE=9,
    GROUND_TILE_TREE_BROWN=10,
    GROUND_TILE_TREE_SNOW=11,
    GROUND_TILE_TREE_DARKGREEN=12,
    GROUND_TILE_TREE_SOLIDBROWN=13,
    GROUND_TILE_TREES=14,
    GROUND_TILE_TREES_2=15,
    GROUND_TILE_FOREST=16,
    GROUND_TILE_HOUSE=17,
    GROUND_TILE_HOUSE_RED=18,
    GROUND_TILE_HOUSE_BLUE=19,
    GROUND_TILE_HOUSE_GREEN=20,
    GROUND_TILE_HOUSE_ORANGE=21,
    GROUND_TILE_HOUSE_PINK=22,
    GROUND_TILE_HOUSE_YELLOW=23,
    GROUND_TILE_HOUSE_TRIPLE=24,
    GROUND_TILE_HOUSE_TRIPLE_RED=25,
    GROUND_TILE_HOUSE_TRIPLE_BLUE=26,
    GROUND_TILE_STREETLIGHT=27,
    STREETLIGHT=27,
    GROUND_TILE_PATH=28,
    WATER_TILE=29,
    MOUNTAIN_TILE=30,
    RAMP_TO_MOUNTAIN=31,
    SNOWY_GROUND_TILE=32,
    SNOWY_GROUND_TILE_TREES_2=33,
    SNOWY_GROUND_TILE_ROCKY=34,
    SNOWY_GROUND_TILE_STONES=35,
    SNOWY_MOUNTAIN_TILE=36,
    SNOWY_RAMP_TO_MOUNTAIN=37,
    CARD_BASE_4=38,
    CARD_BASE_5=39,
    CARD_BASE_6=40,
    MOUNTAIN_TILE_TREE=41,
    SNOWY_MOUNTAIN_TILE_TREE=42,
    GROUND_TILE_STREETLIGHT_FOILAGE=43,
    STREETLIGHT_FOILAGE=43,
    STREETLIGHT_BIG=44,
    STREETLIGHT_BUSHES=45,
    STREETLIGHT_ROCKS=46,
    STREETLIGHT_WIDE=47
)
ID_TO_ASSET = {ASSET_TO_ID[k]: k for k in ASSET_TO_ID}
LANDMARK_NAME_TO_TEXT = dict(
    GROUND_TILE_ROCKY='a rock',
    GROUND_TILE_STONES='a rock',
    GROUND_TILE_STONES_GREENBUSH='a rock',
    GROUND_TILE_STONES_BROWNBUSH='a rock',
    GROUND_TILE_STONES_GREYBUSH='a rock',
    GROUND_TILE_TREE='a tree',
    GROUND_TILE_TREE_BROWN='a tree',
    GROUND_TILE_TREE_SNOW='a tree',
    GROUND_TILE_TREE_DARKGREEN='a tree',
    GROUND_TILE_TREE_SOLIDBROWN='a tree',
    GROUND_TILE_TREES='a clump of trees',
    GROUND_TILE_TREES_2='a tree',
    GROUND_TILE_FOREST='a forest',
    GROUND_TILE_HOUSE='a short grey house',
    GROUND_TILE_HOUSE_RED='a short red house',
    GROUND_TILE_HOUSE_BLUE='a short blue house',
    GROUND_TILE_HOUSE_GREEN='a short green house',
    GROUND_TILE_HOUSE_PINK='a short pink house',
    GROUND_TILE_HOUSE_YELLOW='a short yellow house',
    GROUND_TILE_HOUSE_TRIPLE='a tall grey house',
    GROUND_TILE_HOUSE_TRIPLE_RED='a tall red house',
    GROUND_TILE_HOUSE_TRIPLE_BLUE='a tall blue house',
    STREETLIGHT='a streetlight',
    SNOWY_GROUND_TILE_TREES_2='a tree',
    SNOWY_GROUND_TILE_ROCKY='a rock',
    SNOWY_GROUND_TILE_STONES='a rock',
    SNOWY_MOUNTAIN_TILE_TREE='a tree',
    STREETLIGHT_FOILAGE='a lamppost',
    STREETLIGHT_BIG='a lamppost',
    STREETLIGHT_BUSHES='a lamppost',
    STREETLIGHT_ROCKS='a lamppost',
    STREETLIGHT_WIDE='a lamppost',
)
SHAPES = [
    ('plus', 'plusses'),
    ('circle', 'circles'),
    ('heart', 'hearts'),
    ('diamond', 'diamonds'),
    ('square', 'squares'),
    ('star', 'stars'),
    ('triangle', 'triangles')
]
COLORS = [
    'black',
    'blue',
    'green',
    'orange',
    'pink',
    'red',
    'yellow'
]
NUMBERS = [
    'one',
    'two',
    'three'
]
PREPOSITIONS_NEXT = [
    'next to',
    'beside',
]
PREPOSITIONS_NEAR = [
    'near',
    'close to',
]
PREPOSITIONS_MID = [
    'a few tiles away from',
]
PREPOSITIONS_FAR = [
    "far away from",
    'several tiles away from',
]
HEX_TO_CARTESIAN = np.array(
    [[0.5,            0,          1],
     [np.sqrt(3) / 2, np.sqrt(3), 0]]
)
N_LANDMARKS = 2
N_RUNSETS = 8
N_TRIALS = 12
N_SCENARIOS = 3
MAX_TRIAL_DURATION = 30
MAX_SCENARIO_DURATION = 360
MAX_RUN_DURATION = 100000000


def open_browser(ffservice=None, fullscreen=True):
    from selenium import webdriver
    ffservice = ffservice or webdriver.chrome.service.Service()

    browser = webdriver.Firefox(service=ffservice)

    if fullscreen:
        browser.fullscreen_window()

    return browser


def get_distance(loc1, loc2, manhattan=True):
    a1, r1, c1 = loc1['a'], loc1['r'], loc1['c']
    a2, r2, c2 = loc2['a'], loc2['r'], loc2['c']
    if manhattan:
        U = (c2 - c1) - (r2 - r1)
        V = (a2 - a1) + 2 * (r2 - r1)
        U_s, V_s = np.sign(U), np.sign(V)
        U, V = np.abs(U), np.abs(V)
        if U_s == V_s:
            d = U + V
        else:
            d = np.maximum(U, V)
        d = int(d)
    else:  # Cartesian
        loc1 = np.array([a1, r1, c1])
        loc2 = np.array([a2, r2, c2])
        loc1 = HEX_TO_CARTESIAN @ loc1
        loc2 = HEX_TO_CARTESIAN @ loc2
        d = np.sqrt(((loc1 - loc2) ** 2).sum())

    return d


def create_instructions(scenario, hard=False):
    assets = []
    for asset in scenario['map']['tiles']:
        asset_id = asset['asset_id']
        asset_name = ID_TO_ASSET[asset_id]
        if asset_name in LANDMARK_NAME_TO_TEXT:
            assets.append(asset)
    target_card_ids = [x[0] for x in scenario['target_card_ids']]
    instructions = []
    for target_card_id in target_card_ids:
        instruction = ''
        landmarks = scenario['landmarks'][target_card_id]
        target_card = None
        for card in scenario['prop_update']['props']:
            if card['id'] == target_card_id:
                target_card = card
                break
        assert target_card is not None, 'Target card id %s not found.' % target_card_id
        color_ix = target_card['card_init']['color'] - 1
        shape_ix = target_card['card_init']['shape'] - 1
        count_ix = target_card['card_init']['count'] - 1

        number = NUMBERS[count_ix]
        color = COLORS[color_ix]
        shape = SHAPES[shape_ix][count_ix > 0]

        if instruction:
            instruction += ' Then find'
        else:
            instruction += 'Find'
        instruction += f' the card with {number} {color} {shape}'
        instruction += '{continuation}'

        assets_descriptions_used = set()
        for i, (asset, distance) in enumerate(landmarks):
            asset_id = asset['asset_id']
            asset_name = ID_TO_ASSET[asset_id]
            asset_text = LANDMARK_NAME_TO_TEXT[asset_name]
            if asset_text in assets_descriptions_used:
                already_seen = True
            else:
                already_seen = False
            assets_descriptions_used.add(asset_text)
            if already_seen:
                asset_text = asset_text.replace('an ', 'another ').replace('a ', 'another ')
            # print(asset_name, asset['cell']['coord'], loc_cur, distance)

            if distance == 1:
                preposition = PREPOSITIONS_NEXT[np.random.randint(len(PREPOSITIONS_NEXT))]
            # elif distance == 2:
            #     preposition = PREPOSITIONS_NEAR[np.random.randint(len(PREPOSITIONS_NEAR))]
            else:
                preposition = '%d tiles from' % distance
            if hard and i == 0:  # Object embedding, only at top level RC
                continuation = f' that {asset_text}{{continuation}} is {preposition}'
            else:
                continuation = f' that is {preposition} {asset_text}{{continuation}}'
            instruction = instruction.format(continuation=continuation)

        instruction = instruction.format(continuation='')
        instruction += '.'
        instructions.append(instruction)

    return instructions


def set_difficulty(scenario, task_difficulty=False, linguistic_complexity=False):
    if task_difficulty > 2:
        scenario['map']['fog_start'] = 2
        scenario['map']['fog_end'] = 8
    elif task_difficulty > 1:
        scenario['map']['fog_start'] = 3
        scenario['map']['fog_end'] = 12
    elif task_difficulty > 0:
        scenario['map']['fog_start'] = 4
        scenario['map']['fog_end'] = 16
    else:
        scenario['map']['fog_start'] = 30
        scenario['map']['fog_end'] = 120

    instructions = create_instructions(scenario, hard=linguistic_complexity)
    objectives = []
    for instruction in instructions:
        objective = OBJECTIVE.copy()
        objective['text'] = instruction
        objectives.append(objective)

    scenario['objectives'] = objectives

    return scenario
