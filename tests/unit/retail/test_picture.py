from mloda_demo.retail.picture import depths, detour, lines_of


def test_depth_is_the_longest_path_from_a_root() -> None:
    # 0 -> 1 -> 2 -> 3 and 0 -> 3: node 3 sits after 2, not next to 1.
    assert depths(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == [0, 1, 2, 3]


def test_a_detour_makes_the_direct_arrow_redundant() -> None:
    edges = {(0, 1), (1, 2), (0, 2), (0, 3)}
    assert detour((0, 2), edges)
    assert not detour((0, 3), edges)


def test_a_chained_name_breaks_once_near_its_middle() -> None:
    assert lines_of("line_value__7d_before__last_return") == ["line_value", "__7d_before", "__last_return"]
    assert lines_of("last_return, last_return_value") == ["last_return", "last_return_value"]
