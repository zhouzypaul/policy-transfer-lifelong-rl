"""
based on looking at the game layout in procgen interactive mode, 
I arranged the level-seeds in an order that is roughly increasing in difficulty
"""

caveflyer_levels = [
    0, 9, 11, 14, 15,  # easy: no enemies or rocks, just flying
    5, 8, 2, 12, 10, 16, 17,18, 19, 3, 7, 13,  # medium: enemies and rocks
    1, 4, 6  # hard: enemies that are difficult to avoid or long traversal distance
]


procgen_game_curriculum = {
    'caveflyer': caveflyer_levels,
}