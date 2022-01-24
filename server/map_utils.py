from assets import AssetId
from hex import HecsCoord, HexCell, HexBoundary
from messages.map_update import MapUpdate, Tile
from enum import Enum
import random
import logging
from queue import Queue

logger = logging.getLogger()

def LayerToHeight(layer):
    """ Converts a layer to a height."""
    layer_to_height = {
        0: 0.05,
        1: 0.275,
        2: 0.355,
    }
    if layer not in layer_to_height:
        return layer_to_height[0]

    return layer_to_height[layer]

def EmptyTile():
    return Tile(AssetId.EMPTY_TILE,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0),
                LayerToHeight(0),  # Height (float)
                0,  # Z-Layer (int)
                ),
        0
    )

def GroundTile(rotation_degrees=0):
    """ Creates a single tile of ground."""
    return Tile(
        AssetId.GROUND_TILE,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0),
                LayerToHeight(0),  # Height (float)
                0,  # Z-Layer (int)
                ),
        rotation_degrees
    )


def WaterTile(rotation_degrees=0):
    """ Creates a single tile of Water."""
    return Tile(
        AssetId.WATER_TILE,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0,  # Z-Layer (int)
                ),
        rotation_degrees
    )


def PathTile(rotation_degrees=0):
    """ Creates a single tile of Path."""
    return Tile(
        AssetId.GROUND_TILE_PATH,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0),
                LayerToHeight(0),  # Height (float)
                0,  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileRocky(rotation_degrees=0):
    """ Creates a single tile of rocky ground."""
    return Tile(
        AssetId.GROUND_TILE_ROCKY,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileStones(rotation_degrees=0):
    """ Creates a single tile of ground with stones."""
    return Tile(
        AssetId.GROUND_TILE_STONES,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileTrees(rotation_degrees=0):
    """ Creates a single tile of ground with several trees. """
    return Tile(
        AssetId.GROUND_TILE_TREES,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileSingleTree(rotation_degrees=0):
    """ Creates a single tile of ground with a tree."""
    return Tile(
        AssetId.GROUND_TILE_TREES_2,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileForest(rotation_degrees=0):
    """ Creates a single tile of ground with a forest."""
    return Tile(
        AssetId.GROUND_TILE_FOREST,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )

class HouseType(Enum):
    NONE = 0
    HOUSE = 1
    HOUSE_RED = 2
    HOUSE_BLUE = 3
    TRIPLE_HOUSE = 4
    TRIPLE_HOUSE_RED = 5
    TRIPLE_HOUSE_BLUE = 6
    RANDOM = 7

def AssetIdFromHouseType(type):
    if type == HouseType.HOUSE:
        return AssetId.GROUND_TILE_HOUSE
    elif type == HouseType.HOUSE_RED:
        return AssetId.GROUND_TILE_HOUSE_RED
    elif type == HouseType.HOUSE_BLUE:
        return AssetId.GROUND_TILE_HOUSE_BLUE
    elif type == HouseType.TRIPLE_HOUSE:
        return AssetId.GROUND_TILE_HOUSE_TRIPLE
    elif type == HouseType.TRIPLE_HOUSE_RED:
        return AssetId.GROUND_TILE_HOUSE_TRIPLE_RED
    elif type == HouseType.TRIPLE_HOUSE_BLUE:
        return AssetId.GROUND_TILE_HOUSE_TRIPLE_BLUE
    elif type == HouseType.RANDOM:
        return random.choice([AssetId.GROUND_TILE_HOUSE,
                              AssetId.GROUND_TILE_HOUSE_RED,
                              AssetId.GROUND_TILE_HOUSE_BLUE,
                              AssetId.GROUND_TILE_HOUSE_TRIPLE,
                              AssetId.GROUND_TILE_HOUSE_TRIPLE_RED,
                              AssetId.GROUND_TILE_HOUSE_TRIPLE_BLUE])
    else:
        logger.error(f"Unknown house type: {type}")
        return AssetId.HOUSE

def GroundTileHouse(rotation_degrees=0, type=HouseType.HOUSE):
    """ Creates a single tile of ground with a house."""
    asset_id = AssetIdFromHouseType(type)
    return Tile(
        asset_id,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def GroundTileStreetLight(rotation_degrees=0):
    """ Creates a single tile of ground with a street light."""
    return Tile(
        AssetId.GROUND_TILE_STREETLIGHT,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0x3F),
                LayerToHeight(0),  # Height (float)
                0  # Z-Layer (int)
                ),
        rotation_degrees
    )


def MountainTile(rotation_degrees=0):
    """ Creates a single tile of mountain."""
    return Tile(
        AssetId.MOUNTAIN_TILE,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary(0),
                LayerToHeight(2),  # Height (float)
                2  # Z-Layer (int)
                ),
        rotation_degrees
    )


def RampToMountain(rotation_degrees=0):
    """ Creates a single tile of ramp."""
    return Tile(
        AssetId.RAMP_TO_MOUNTAIN,
        HexCell(HecsCoord.from_offset(0, 0), HexBoundary.rotate_cw(HexBoundary(0b101101), rotation_degrees),
                LayerToHeight(1),  # Height (float)
                1  # Z-Layer (int)
                ),
        rotation_degrees
    )

# Tile boundaries must prevent leaving the map, or undefined behavior will occur.
def FloodFillPartitionTiles(tiles):
    tile_by_loc = {}
    for tile in tiles:
        tile_by_loc[tile.cell.coord] = tile
    
    unvisited_tiles = set(tiles)
    visited_tiles = set()
    tile_queue = Queue()
    partitions = []

    while len(unvisited_tiles) > 0:
        tile = unvisited_tiles.pop()
        tile_queue.put(tile)
        partition = []
        # Do a floodfill to create a partition on the map from this tile.
        while not tile_queue.empty():
            tile = tile_queue.get()
            if tile in visited_tiles:
                continue
            partition.append(tile)
            if tile in unvisited_tiles:
                unvisited_tiles.remove(tile)
            visited_tiles.add(tile)
            coord = tile.cell.coord
            boundary = tile.cell.boundary
            # Add all neighbors that aren't blocked by an edge.
            for neighbor in coord.neighbors():
                if neighbor not in tile_by_loc:
                    continue
                if not boundary.get_edge_between(coord, neighbor):
                    tile_queue.put(tile_by_loc[neighbor])
        partitions.append(partition)
    return partitions