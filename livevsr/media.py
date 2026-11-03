"""Local-only FFmpeg I/O; temporary decoded media is cleaned by the CLI."""
import json
import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                              '-of', 'json', str(path)]))['streams']


def decode_video(path, destination, scale):
    if Path(path).is_dir():
        paths = sorted(Path(path).glob('*.png'))
        if not paths:
            raise ValueError('Input directory has no PNG frames')
        with Image.open(paths[0]) as image:
            width, height = image.size
        for frame in paths:
            with Image.open(frame) as image:
                if image.size != (width, height):
                    raise ValueError('All input frames must have the same dimensions')
        if width != height or width * scale % 32:
            raise ValueError('Use square frames with output size divisible by 32')
        return paths, width * scale, height * scale
    streams = probe(path)
    video = next(s for s in streams if s['codec_type'] == 'video')
    width, height = int(video['width']), int(video['height'])
    if width != height or width * scale % 32:
        raise ValueError("Use an already-cropped square face video with output size divisible by 32")
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-vf', 'fps=25', '-threads', '2', str(destination / '%08d.png')], check=True)
    paths = sorted(destination.glob('*.png'))
    if not paths:
        raise ValueError('Input video has no decodable frames')
    return paths, width * scale, height * scale


def decode_scene(path, destination):
    """Keep native CFR frames for DWPose; resample geometry and RGB together later."""
    if Path(path).is_dir():
        paths = sorted(Path(path).glob('*.png'))
        fps = Fraction(25)
    else:
        video = next(s for s in probe(path) if s['codec_type'] == 'video')
        fps = Fraction(video['avg_frame_rate'])
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
            '-map', '0:v:0', '-vsync', '0', '-threads', '2',
            str(destination / '%08d.png')], check=True)
        paths = sorted(destination.glob('*.png'))
    if not paths:
        raise ValueError('Input has no decodable frames')
    with Image.open(paths[0]) as image:
        width, height = image.size
    if width % 2 or height % 2:
        raise ValueError('Full-scene H.264 output requires even image dimensions')
    return paths, width, height, fps


def prepare_audio(source, destination, frames):
    if not any(s['codec_type'] == 'audio' for s in probe(source)):
        raise ValueError('No audio stream; supply --audio with the original or target speech')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-map', '0:a:0',
        '-af', 'apad', '-t', str(frames / 25), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
        str(destination)], check=True)
    with wave.open(str(destination), 'rb') as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
    needed = frames * 640
    return np.pad(pcm[:needed], (0, max(0, needed - len(pcm)))).reshape(frames, 640)


def write_audio(path, pcm):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())


def encode_video(frames, audio, destination, *, size=None, count=None):
    width, height = size if size is not None else frames.shape[2:0:-1]
    count = len(frames) if count is None else count
    process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25', '-i', 'pipe:0',
        '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '-1',
        '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', '-t', str(count / 25), '-movflags', '+faststart',
        str(destination)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f'FFmpeg encoding failed ({code})')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(destination),
                    '-f', 'null', '-'], check=True)


def check_tools():
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise RuntimeError(f'{executable} must be installed and on PATH')
