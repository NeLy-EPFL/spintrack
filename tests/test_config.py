import textwrap

from spintrack.config import Config

SAMPLE = textwrap.dedent(
    """
    ## FicTrac v2.1.2 config file (build date Oct 25 2023)
    c2a_cnrs_xy      : { 191, 171, 128, 272, 20, 212, 99, 132 }
    c2a_r            : { 0.722445, -0.131314, -0.460878 }
    c2a_src          : c2a_cnrs_xy
    do_display       : y
    opt_do_global    : n
    q_factor         : 6
    roi_circ         : { 63, 171, 81, 145, 106, 135, 150, 160 }
    roi_ignr         : { { 96, 156, 113, 147, 106, 128 }, { 71, 213, 90, 219, 114, 218 } }
    src_fn           : sample.mp4
    thr_ratio        : 1.25
    vfov             : 45
    accumulate_map   : n
    my_custom_key    : hello
    # a comment kept for the user
    """
)


def test_parse_fictrac_text():
    cfg = Config.from_text(SAMPLE)
    assert cfg.vfov == 45.0 and isinstance(cfg.vfov, float)
    assert cfg.q_factor == 6 and cfg.do_display is True and cfg.opt_do_global is False
    assert cfg.roi_circ == [63, 171, 81, 145, 106, 135, 150, 160]
    assert cfg.roi_ignr == [[96, 156, 113, 147, 106, 128], [71, 213, 90, 219, 114, 218]]
    assert cfg.c2a_r == [0.722445, -0.131314, -0.460878]
    assert cfg.accumulate_map is False
    assert cfg.extra == {"my_custom_key": "hello"}
    assert cfg.comments == ["# a comment kept for the user"]
    assert cfg.window_size() == 60 and cfg.has_ball()


def test_text_round_trip_preserves_values(tmp_path):
    cfg = Config.from_text(SAMPLE)
    path = tmp_path / "config.txt"
    cfg.save(path)
    again = Config.load(path)
    assert again.to_mapping() == cfg.to_mapping()
    assert "my_custom_key    : hello" in path.read_text()


def test_yaml_round_trip(tmp_path):
    cfg = Config.from_text(SAMPLE)
    path = tmp_path / "config.yaml"
    cfg.save(path)
    again = Config.load(path)
    assert again.roi_ignr == cfg.roi_ignr and again.vfov == 45.0
    assert again.extra == {"my_custom_key": "hello"}


def test_defaults_follow_fictrac():
    cfg = Config()
    assert cfg.vfov is None and cfg.q_factor == 6 and cfg.opt_bound == 0.35
    assert cfg.thr_ratio == 1.25 and cfg.sock_port == -1 and cfg.accumulate_map is True
    assert not cfg.has_ball()


def test_real_lab_configs_parse(lab_configs):
    for path in lab_configs:
        cfg = Config.load(path)
        assert cfg.vfov is not None and cfg.has_ball()
        assert Config.from_text(cfg.to_text()).to_mapping() == cfg.to_mapping()
