﻿using System;
using UnityEngine;

// Interface for loading assets.
public interface IAssetSource
{
    public enum AssetId
    {
        PLAYER,
        PLAYER_WITH_CAM,
        FOLLOWER_BOT,
        GROUND_TILE,
        GROUND_TILE_ROCKY,
        GROUND_TILE_STONES,
        GROUND_TILE_STONES_GREENBUSH,
        GROUND_TILE_STONES_BROWNBUSH,
        GROUND_TILE_STONES_GREYBUSH,
        GROUND_TILE_TREE,
        GROUND_TILE_TREE_BROWN,
        GROUND_TILE_TREE_SNOWY,
        GROUND_TILE_TREE_DARK_GREEN,
        GROUND_TILE_TREE_SOLID_BROWN,
        GROUND_TILE_TREES,
        GROUND_TILE_TREES_2,
        GROUND_TILE_FOREST,
        GROUND_TILE_HOUSE,
        GROUND_TILE_HOUSE_RED,
        GROUND_TILE_HOUSE_BLUE,
        GROUND_TILE_HOUSE_GREEN,
        GROUND_TILE_HOUSE_ORANGE,
        GROUND_TILE_HOUSE_PINK,
        GROUND_TILE_HOUSE_YELLOW,

        GROUND_TILE_HOUSE_TRIPLE,
        GROUND_TILE_HOUSE_TRIPLE_RED,
        GROUND_TILE_HOUSE_TRIPLE_BLUE,
        GROUND_TILE_STREETLIGHT,
        GROUND_TILE_PATH,
        WATER_TILE,
        MOUNTAIN_TILE,
        RAMP_TO_MOUNTAIN,
        SNOWY_GROUND_TILE,
        SNOWY_GROUND_TILE_TREES_2,
        SNOWY_GROUND_TILE_ROCKY,
        SNOWY_GROUND_TILE_STONES,
        SNOWY_MOUNTAIN_TILE,
        SNOWY_RAMP_TO_MOUNTAIN,
        CARD_BASE_1,
        CARD_BASE_2,
        CARD_BASE_3,
        MOUNTAIN_TREE,
        SNOWY_MOUNTAIN_TREE,
        GROUND_TILE_STREETLIGHT_FOILAGE,
        STREETLIGHT_BIG,
        STREETLIGHT_BUSHES,
        STREETLIGHT_ROCKS,
        STREETLIGHT_WIDE,
        // These are 2D shapes that appear on card faces.
        SQUARE,
        STAR,
        TORUS,
        TRIANGLE,
        PLUS,
        HEART,
        DIAMOND,
        // Used for indicating a location on the ground (in tutorials).
        GROUND_PULSE_INDICATOR_YELLOW,
        TUTORIAL_INDICATOR,
        NONE = 101,
        MAX,
        // Aliases go here.
        STREETLIGHT = GROUND_TILE_STREETLIGHT,
        STREETLIGHT_FOILAGE = GROUND_TILE_STREETLIGHT_FOILAGE,
    }

    public enum MaterialId
    {
        CARD_BACKGROUND,
        // These are colors for the 2D shapes on card faces.
        COLOR_BLACK,
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_ORANGE,
        COLOR_PINK,
        COLOR_RED,
        COLOR_YELLOW,
        // This material is emissive and used to draw the card's outline.
        CARD_OUTLINE,
    }

    public enum UiId
    {
        OBJECTIVE_ACTIVE,
        OBJECTIVE_COMPLETE,
        OBJECTIVE_PENDING,
        OBJECTIVE_CANCELLED,
        MENU_BUTTON,
        OBJECTIVE_FEEDBACK,
    }

    // Returns a prefab of the requested asset.
    GameObject Load(AssetId assetId);

    // Loads a material into memory.
    Material LoadMat(MaterialId material);

    GameObject LoadUi(UiId uiId);

    // TODO(sharf): If we want to remove unity-specific interfaces entirely,
    // this interface can be rewritten to add something similar to Unity's
    // "Instantiate" function. 
}