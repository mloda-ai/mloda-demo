from mloda_demo.physical_ai import welded
from mloda_demo.physical_ai.readers import DepthPng, TumDepth
from mloda_demo.physical_ai.runner import run
from mloda_demo.physical_ai.trace import chain, format_trace, trace_html, trace_lines


def test_trace_follows_the_data_flow_and_marks_the_undeclared_scale() -> None:
    lines = trace_lines(run(DepthPng))
    assert [line.role for line in lines] == ["reader", "convert", "nearest", "stop"]
    reader, convert, nearest, stop = lines
    assert (reader.step, reader.context, reader.detail) == (
        "DepthPng",
        "src pioneer_slam/depth/053.png",
        "raw 2915 (uint16, scale undeclared)",
    )
    assert convert.detail == "assumes mm: 2915 x 0.001 = 2.92 m"
    assert (nearest.context, nearest.detail) == ("unit m", "frame 53: 2.92 m")
    assert (stop.context, stop.detail) == ("threshold 1 m", "go")
    assert (reader.marks, convert.marks) == (("scale undeclared",), ("assumes mm",))


def test_declared_reader_has_nothing_to_mark() -> None:
    lines = trace_lines(run(TumDepth))
    assert lines[0].detail == "raw 2915 (uint16, scale 5000 per m)"
    assert lines[1].detail == "scale 5000 per m: 2915 / 5000 = 0.58 m"
    assert lines[3].detail == "stop"
    assert all(not line.marks for line in lines)
    assert lines[0].version != trace_lines(run(DepthPng))[0].version


def test_trace_can_show_any_frame() -> None:
    lines = trace_lines(run(TumDepth), frame=0)
    assert lines[2].detail == "frame 0: 1.85 m"
    assert lines[3].detail == "go"


def test_rendered_trace_highlights_the_scale_and_the_assumption() -> None:
    result = run(DepthPng)
    header = format_trace(result).splitlines()[0]
    assert header.startswith("run ") and "generic PNG · frame 53" in header
    rendered = trace_html(result)
    assert "<mark>scale undeclared</mark>" in rendered
    assert "<mark>assumes mm</mark>" in rendered


def test_failed_run_has_no_trace_lines() -> None:
    assert trace_lines(run(DepthPng, check=True)) == []


def test_chain_names_the_reader_and_welded_runs_share_nothing() -> None:
    assert chain(run(DepthPng)) == ["DepthPng", "DepthToMetres", "NearestAhead", "StopRule"]
    assert chain(run(TumDepth)) == ["TumDepth", "DepthToMetres", "NearestAhead", "StopRule"]
    kinect = chain(run(None, features=welded.KINECT_FEATURES, feature_groups=welded.KINECT_CHAIN))
    tum = chain(run(None, features=welded.TUM_FEATURES, feature_groups=welded.TUM_CHAIN))
    assert kinect == ["KinectDepthPng", "KinectToMetres", "KinectNearest", "KinectStop"]
    assert not set(kinect) & set(tum)
