from mloda_demo.retail.picture import depths, detour


def test_depth_is_the_longest_path_from_a_root() -> None:
    # 0 -> 1 -> 2 -> 3 and 0 -> 3: node 3 sits after 2, not next to 1.
    assert depths(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == [0, 1, 2, 3]


def test_a_detour_makes_the_direct_arrow_redundant() -> None:
    edges = {(0, 1), (1, 2), (0, 2), (0, 3)}
    assert detour((0, 2), edges)
    assert not detour((0, 3), edges)
