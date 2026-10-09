import cv2
import numpy as np

from spintrack.io.sources import VideoSource, open_source


def _write_video(path, n=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 20 * i, np.uint8)
        writer.write(frame)
    writer.release()


def test_video_source_yields_gray_frames_with_timestamps(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path)
    src = open_source(path)
    assert isinstance(src, VideoSource) and (src.width, src.height) == (64, 48)
    frames = list(src)
    src.close()
    assert len(frames) == 10
    assert frames[0].image.dtype == np.uint8 and frames[0].image.shape == (48, 64)
    assert frames[-1].ts_ms > frames[0].ts_ms and frames[3].index == 3
    assert abs(int(frames[5].image.mean()) - 100) < 6


def test_an_avi_with_b_frames_is_timed_and_sought_by_frame_count(tmp_path):
    """AVI frames are evenly timed and seeks land on the frame asked for.

    AVI keeps no presentation times, and FFmpeg's guesses scramble with B-frames (a
    lab's camera writes such files); seeking lands where decoding through to it does.
    """
    import av
    import pytest

    path = tmp_path / "clip.avi"
    with av.open(str(path), "w", format="avi") as out:
        try:
            stream = out.add_stream("libx264", rate=50)
        except av.error.FFmpegError, ValueError:
            pytest.skip("no H.264 encoder")
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"bf": "2", "x264-params": "keyint=25:scenecut=0"}
        for i in range(80):
            image = np.zeros((48, 64, 3), np.uint8)
            image[:, i % 64] = 255  # a bar that moves one column a frame
            image[..., 2] = 3 * i
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, "rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    src = VideoSource(path)
    frames = list(src)
    src.close()
    assert [f.ts_ms for f in frames[:5]] == [0.0, 20.0, 40.0, 60.0, 80.0]
    for index in (10, 30, 61):
        src = VideoSource(path)
        src.seek(index)
        frame = src.read()
        src.close()
        assert frame.index == index
        assert np.array_equal(frame.image, frames[index].image)
