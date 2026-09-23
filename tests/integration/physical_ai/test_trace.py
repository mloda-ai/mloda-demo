from mloda_demo.physical_ai.readers import DepthReaderA, DepthReaderB
from mloda_demo.physical_ai.runner import FEATURES, run
from mloda_demo.physical_ai.trace import format_trace, trace_html, trace_lines


def test_trace_follows_the_data_flow_and_marks_the_unit_mismatch() -> None:
    lines = trace_lines(run(DepthReaderB, label="device B", fault=True))
    assert [line.role for line in lines] == ["reader", "convert", "calibrate", "distance", "brake"]
    reader, convert, calibrate, distance, brake = lines
    assert (reader.step, reader.context, reader.detail) == (
        "DepthReaderB",
        "src device_b.npz",
        "raw 1200 (uint16, unit mm)",
    )
    assert convert.detail == "assumes cm: 1200 x 0.01 = 12 m"
    assert (calibrate.context, calibrate.detail) == ("calib 2026-09-01", "frame camera_optical -> base_link")
    assert (distance.context, distance.detail) == ("unit m", "object 7: 12 m")
    assert (brake.context, brake.detail) == ("threshold 2 m", "object 7: no brake")
    assert (reader.marks, convert.marks) == (("unit mm",), ("assumes cm",))


def test_fixed_trace_has_nothing_to_mark() -> None:
    lines = trace_lines(run(DepthReaderB))
    assert lines[1].detail == "assumes mm: 1200 x 0.001 = 1.2 m"
    assert all(not line.marks for line in lines)
    assert lines[0].version != trace_lines(run(DepthReaderA))[0].version


def test_trace_adds_a_line_for_a_new_feature() -> None:
    lines = trace_lines(run(DepthReaderA, features=(*FEATURES, "approaching")))
    assert (lines[-1].role, lines[-1].detail) == ("approach", "approaching: 3")


def test_rendered_trace_highlights_both_units() -> None:
    result = run(DepthReaderB, label="device B", fault=True)
    assert format_trace(result).splitlines()[0].endswith("device_b · device B")
    rendered = trace_html(result)
    assert "<mark>unit mm</mark>" in rendered
    assert "<mark>assumes cm</mark>" in rendered


def test_failed_run_has_no_trace_lines() -> None:
    assert trace_lines(run(DepthReaderB, fault=True, check_units=True)) == []
