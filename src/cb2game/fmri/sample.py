from datetime import timedelta, time
import string
import logging
import json
import sys
import os
import subprocess
import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cb2game.pyclient.remote_client import RemoteClient
from cb2game.pyclient.game_endpoint import Action
from cb2game.fmri.utils import N_LANDMARKS, ID_TO_ASSET, LANDMARK_NAME_TO_TEXT, N_RUNSETS, N_TRIALS, MAX_TRIAL_DURATION,\
    open_browser, get_distance, set_difficulty


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

HZ = 10
HOST = 'http://localhost:8080'
LOBBY = 'open'
MAX_CARDS = None
N_TARGETS = 25
N_DISTRACTORS = 2
CARD_COVERS = True


def clean_scenario(scenario, card_covers=CARD_COVERS):
    scenario['scenario_id'] = 'scenario'
    scenario['prop_update'] = dict(props=scenario['props'])
    scenario['map'] = scenario['map_update']
    scenario['actor_state'] = dict(actors=scenario['actors'])
    del scenario['props']
    del scenario['map_update']
    del scenario['instructions']
    del scenario['live_feedback']
    del scenario['actors']

    scenario['actor_state']['population'] = 2
    scenario['actor_state']['player_id'] = 21
    scenario['actor_state']['player_role'] = 2

    scenario['turn_state']['turn'] = 1
    scenario['turn_state']['moves_remaining'] = 1000000
    scenario['turn_state']['turns_left'] = 0

    objectives = [dict(
        sender=2,
        text='',
        uuid='',
        completed=False,
        cancelled=False
    )]

    scenario['objectives'] = objectives

    for card in scenario['prop_update']['props']:
        if card_covers:
            card['card_init']['hidden'] = False
        else:
            card['card_init']['hidden'] = True

    for actor in scenario['actor_state']['actors']:
        if actor['actor_role'] == 2:
            actor['location'] = dict(
                a=100,
                r=100,
                c=100
            )
        elif actor['actor_role'] == 1:
            actor['asset_id'] = 2

    scenario['duration_s'] = MAX_TRIAL_DURATION

    return scenario


def sample_scenario(
        browser,
        host=HOST,
        lobby=LOBBY
):
    suffix = "&auto=join_game_queue"
    url = f"{host}/play?lobby_name={lobby}"
    browser.get(url + suffix)
    logger.info(f"Trying to connect to {host} and lobby {lobby}")
    client = RemoteClient(url=host, render=False, lobby_name=lobby)
    connected, reason = client.Connect()
    logger.info(f"Client connected: {connected}")

    game, _ = client.JoinGame()
    logger.info(f"Attached scenario.")
    scenario = game._state().to_dict()
    game.close()

    scenario = clean_scenario(scenario)
    scenario = resample_scenario(scenario)

    return scenario


def sample_card_properties():
    return dict(
        color=np.random.randint(1, 7),
        shape=np.random.randint(1, 7),
        count=np.random.randint(1, 3),
        selected=0,
        hidden=False
    )


def sample_landmarks(n_landmarks, target_card, assets, available_cards, temp=1):
    target_loc = target_card['prop_info']['location']
    landmarks = []
    excluded_distractor_card_ids = set()

    assets_used = []
    assets_by_type = {}
    for asset in assets:
        if asset['asset_id'] not in assets_by_type:
            assets_by_type[asset['asset_id']] = []
        assets_by_type[asset['asset_id']].append(asset)

    for i in range(n_landmarks):
        _assets = [
            dict(asset=x, d=get_distance(x['cell']['coord'], target_loc)) for x in assets if x not in assets_used
        ]
        _assets = sorted(_assets, key=lambda x: x['d'], reverse=False)
        probs = 1 / np.exp(np.arange(len(_assets)) ** (1 / temp))
        probs /= probs.sum()
        s = np.random.choice(_assets, p=probs)
        landmark, distance = s['asset'], s['d']
        landmarks.append((landmark, distance))
        assets_used.append(landmark)
        if i == 0:  # Find and exclude any other cards that happen to match the landmarks
            for card_id in available_cards:
                reference_loc = available_cards[card_id]['prop_info']['location']
                match = False
                for landmark, d in landmarks:
                    asset_type = landmark['asset_id']
                    candidates = assets_by_type[asset_type]
                    match = False
                    for candidate in candidates:
                        if candidate != landmark:
                            _d = get_distance(reference_loc, candidate['cell']['coord'])
                            if d == _d:
                                match = True
                                break
                    if not match:
                        break
                    reference_loc = landmark['cell']['coord']
                if match:
                    excluded_distractor_card_ids.add(card_id)

        target_loc = landmark['cell']['coord']

    return landmarks, excluded_distractor_card_ids


def resample_scenario(
        scenario,
        n_targets=N_TARGETS,
        n_distractors=N_DISTRACTORS,
        n_landmarks=N_LANDMARKS,
        max_cards=MAX_CARDS
):
    cards = scenario['prop_update']['props']
    assets = []
    for asset in scenario['map']['tiles']:
        asset_id = asset['asset_id']
        asset_name = ID_TO_ASSET[asset_id]
        if asset_name in LANDMARK_NAME_TO_TEXT:
            assets.append(asset)

    if max_cards and len(cards) > max_cards:
        cards = np.random.choice(cards, size=max_cards, replace=False)
    cards = list(cards)

    scenario['prop_update']['props'] = list(cards)

    _cards = cards[1:]  # For some reason selection borders are buggy when card 0 is a target, so exclude
    target_set = np.random.choice(_cards, size=n_targets, replace=False)
    targets = {card['id']: card for card in target_set}
    target_ids = [int(x) for x in np.random.permutation(list(targets.keys()))]
    target_properties = {}
    available_cards = {x['id']: x for x in cards if x['id'] not in target_ids}

    landmarks = {}
    excluded_distractor_card_ids = {}
    for target_id in target_ids:
        target_card = targets[target_id]
        _target_properties = sample_card_properties()
        target_properties[target_id] = _target_properties
        landmarks[target_id], excluded_distractor_card_ids[target_id] = sample_landmarks(
            n_landmarks,
            target_card,
            assets,
            available_cards
        )
        for card in cards:  # Give the target card its assigned properties
            if card['id'] == target_id:
                card['card_init'] = _target_properties
            else:  # No other card can have the target's properties (for now)
                if card['card_init'] == _target_properties:
                    nontarget_properties = _target_properties
                    while nontarget_properties in target_properties.values():
                        nontarget_properties = sample_card_properties()
                    card['card_init'] = nontarget_properties

    for target_id in target_ids:
        _target_properties = target_properties[target_id]
        _available_ids = list(set(list(available_cards.keys())) - excluded_distractor_card_ids[target_id])
        distractor_ids = np.random.choice(_available_ids, size=n_distractors, replace=False)
        distractor_ids = list(distractor_ids)
        for distractor_id in distractor_ids:
            distractor = available_cards.pop(distractor_id)
            distractor['card_init'] = _target_properties

    scenario['target_card_ids'] = [[x] for x in target_ids]
    scenario['landmarks'] = landmarks

    return scenario


def sample_runset(
        name,
        browser,
        n=N_TRIALS,
        host=HOST,
        lobby=LOBBY,
        outdir='scenarios_sampled',
        overwrite=False
):
    for i in range(n):
        logger.info(f"Sampling {name}, trial {i + 1}")
        runset_path = os.path.join(outdir, name)
        if not os.path.exists(runset_path):
            os.makedirs(runset_path)
        execute = True
        if not overwrite and (
                    os.path.exists(os.path.join(runset_path, 'scenario_%04d_t0_l0.json' % (i + 1))) and
                    os.path.exists(os.path.join(runset_path, 'scenario_%04d_t0_l1.json' % (i + 1))) and
                    os.path.exists(os.path.join(runset_path, 'scenario_%04d_t1_l0.json' % (i + 1))) and
                    os.path.exists(os.path.join(runset_path, 'scenario_%04d_t1_l1.json' % (i + 1)))
               ):  # Skip execution only if not overwrite and all expected output files are present
            execute = False
        if execute:
            scenario = sample_scenario(browser, host=host, lobby=lobby)
            for task_difficulty in range(2):
                for linguistic_complexity in range(2):
                    scenario_id = 'scenario_%04d_t%d_l%d' % (i + 1, task_difficulty, linguistic_complexity)
                    scenario['scenario_id'] = 'trial_id'
                    scenario = set_difficulty(scenario, task_hard=task_difficulty, lang_hard=linguistic_complexity)
                    path = os.path.join(runset_path, '%s.json' % scenario_id)
                    with open(path, 'w') as f:
                        json.dump(scenario, f, indent=2)


def main(
        browser,
        n_runsets=N_RUNSETS,
        n_trials=N_TRIALS,
        host=HOST,
        lobby=LOBBY,
        outdir='materials',
        overwrite=False
):
    for code in list(string.ascii_uppercase)[:n_runsets]:
        name = 'runset_%s' % code
        sample_runset(
            name,
            browser,
            n=n_trials,
            host=host,
            lobby=lobby,
            outdir=outdir,
            overwrite=overwrite
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("fmri")
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--lobby", type=str, default=LOBBY)
    parser.add_argument("--outdir", type=str, default='materials')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    host = args.host
    lobby = args.lobby

    browser = open_browser(fullscreen=False)

    excp = None
    try:
        main(browser, host=host, lobby=lobby, outdir=args.outdir, overwrite=args.overwrite)
    except Exception as e:
        excp = e

    browser.close()

    if excp is not None:
        raise excp