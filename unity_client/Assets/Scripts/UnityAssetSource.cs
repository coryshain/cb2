using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityAssetSource : IAssetSource
{
    // Maps IAssetSource.AssetId to resource paths in Unity.
    // Must be kept in order with the enum definitions in IAssetSource.
    private static readonly string[] assetPaths = new string[]{
        "Prefab/Actors/Player",
        "Prefab/Actors/PlayerWithCam",
        "Prefab/Actors/CubeRobot",
        "Prefab/Tiles/GroundTile_1",
        "Prefab/Tiles/GroundTile_Rocky_1",
        "Prefab/Tiles/GroundTile_Stones_1",
        "Prefab/Tiles/GroundTile_Stones_1_greenbush",
        "Prefab/Tiles/GroundTile_Stones_1_brownbush",
        "Prefab/Tiles/GroundTile_Stones_1_greybush",
        "Prefab/Tiles/GroundTile_Tree",
        "Prefab/Tiles/GroundTile_Tree_Brown",
        "Prefab/Tiles/GroundTile_Tree_Snow",
        "Prefab/Tiles/GroundTile_Tree_DarkGreen",
        "Prefab/Tiles/GroundTile_Tree_SolidBrown",
        "Prefab/Tiles/GroundTile_Trees_1",
        "Prefab/Tiles/GroundTile_Trees_2",
        "Prefab/Tiles/GroundTile_Forest",
        "Prefab/Tiles/GroundTile_House",
        "Prefab/Tiles/GroundTile_House_red",
        "Prefab/Tiles/GroundTile_House_blue",
        "Prefab/Tiles/GroundTile_House_green",
        "Prefab/Tiles/GroundTile_House_orange",
        "Prefab/Tiles/GroundTile_House_pink",
        "Prefab/Tiles/GroundTile_House_yellow",
        "Prefab/Tiles/GroundTile_TripleHouse",
        "Prefab/Tiles/GroundTile_TripleHouse_red",
        "Prefab/Tiles/GroundTile_TripleHouse_blue",
        "Prefab/Tiles/GroundTile_StreetLight",
        "Prefab/Tiles/PathTile",
        "Prefab/Tiles/WaterTile",
        "Prefab/Mountain/M_Mountain_0Side",
        "Prefab/Tiles/RampTile",
        "Prefab/Tiles/Snowy_GroundTile_1",
        "Prefab/Tiles/Snowy_GroundTile_Trees_2",
        "Prefab/Tiles/Snowy_GroundTile_Rocky_1",
        "Prefab/Tiles/Snowy_GroundTile_Stones_1",
        "Prefab/Mountain/Snowy_M_Mountain_0Side",
        "Prefab/Tiles/Snowy_RampTile",
        "Prefab/Cards/CardBase_1",
        "Prefab/Cards/CardBase_2",
        "Prefab/Cards/CardBase_3",
        "Prefab/Mountain/Mountain_Tree",
        "Prefab/Mountain/Snowy_Mountain_Tree",
        "Prefab/Tiles/GroundTile_StreetLight_Foilage",
        "Prefab/Tiles/GroundTile_StreetLight_Big",
        "Prefab/Tiles/GroundTile_StreetLight_Bushes",
        "Prefab/Tiles/GroundTile_StreetLight_Rocks",
        "Prefab/Tiles/GroundTile_StreetLight_Wide",
        // These are 2D shapes that appear on card faces.
        "Prefab/Cards/Shapes/Square",
        "Prefab/Cards/Shapes/Star",
        "Prefab/Cards/Shapes/Torus",
        "Prefab/Cards/Shapes/Triangle",
        "Prefab/Cards/Shapes/Plus",
        "Prefab/Cards/Shapes/Heart",
        "Prefab/Cards/Shapes/Diamond",
        // Used for indicating a location on the ground (in tutorials).
        "Prefab/ObjectGroups/GroundPulse_yellow",
        "Prefab/ObjectGroups/tutorial_indicator",
    };

    // Maps IAssetSource.MaterialId to resource paths in Unity.
    // Must be kept in order with the enum definitions in IAssetSource.
    private static readonly string[] materialPaths = new string[] {
        "Prefab/Cards/Materials/Card",
        "Prefab/Cards/Materials/card_black",
        "Prefab/Cards/Materials/card_blue",
        "Prefab/Cards/Materials/card_green",
        "Prefab/Cards/Materials/card_orange",
        "Prefab/Cards/Materials/card_pink",
        "Prefab/Cards/Materials/card_red",
        "Prefab/Cards/Materials/card_yellow",
        "Prefab/Cards/Materials/card_outline",
    };

    private static Logger _logger = Logger.GetOrCreateTrackedLogger("UnityAssetSource");

    // Maps IAssetSource.UiId to resource paths in Unity.
    // Must be kept in order with the enum definitions in IAssetSource.
    private static readonly string[] uiPaths = new string[] {
        "Prefab/UI/Instruction_Prefabs/ActiveObjective",
        "Prefab/UI/Instruction_Prefabs/CompletedObjective",
        "Prefab/UI/Instruction_Prefabs/PendingObjective",
        "Prefab/UI/Instruction_Prefabs/CancelledObjective",
        "Prefab/UI/MenuButton",
        "Prefab/UI/Instruction_Prefabs/FeedbackLine",
    };

    public GameObject Load(IAssetSource.AssetId assetId)
    {
        int assetIndex = (int)assetId;
        if (assetIndex >= assetPaths.Length)
        {
            _logger.Info("Asset index out of range: " + assetIndex);
            return null;
        }
        GameObject obj = Resources.Load<GameObject>(assetPaths[assetIndex]);
        if (obj == null)
        {
            _logger.Info("Null: " + assetPaths[assetIndex]);
        }
        return obj;
    }

    public Material LoadMat(IAssetSource.MaterialId materialId)
    {
        int materialIndex = (int)materialId;
        if (materialIndex >= materialPaths.Length)
        {
            _logger.Info("Material index out of range: " + materialIndex);
            return null;
        }
        Material mat = Resources.Load<Material>(materialPaths[materialIndex]);
        if (mat == null)
        {
            _logger.Info("Null: " + materialPaths[materialIndex]);
        }
        return mat;
    }
    public GameObject LoadUi(IAssetSource.UiId uiId)
    {
        int uiIndex = (int)uiId;
        if (uiIndex >= uiPaths.Length)
        {
            _logger.Info("UI index out of range: " + uiIndex);
            return null;
        }
        GameObject obj = Resources.Load<GameObject>(uiPaths[uiIndex]);
        if (obj == null)
        {
            _logger.Info("Null: " + assetPaths[uiIndex]);
        }
        return obj;
    }
}