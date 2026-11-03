"""77-frame windows: 5 bootstrap frames, 72 payload frames, 16-frame overlap."""


def windows(frame_count):
    if frame_count < 1:
        raise ValueError("Video contains no frames")
    result, start, retained = [], 0, 0
    while retained < frame_count:
        requested = [0] * 5 + list(range(72)) if not result else list(range(start - 5, start + 72))
        stop = min(start + 72, frame_count)
        begin = 0 if not result else start + 16
        emitted = stop - begin
        decode_start = 0 if not result else 6
        decode_stop = 20 if not result else decode_start + (emitted + 3) // 4
        result.append(dict(input_indices=[min(i, frame_count - 1) for i in requested],
            latent_start=start // 4, retained_start=begin, retained_stop=stop,
            emit_frames=emitted, decode_start=decode_start, decode_stop=decode_stop,
            discard=5 if not result else 0))
        retained, start = stop, start + 56
    return result


def input_indices(frame_count):
    last_start = windows(frame_count)[-1]['latent_start'] * 4
    return [0] * 5 + [min(i, frame_count - 1) for i in range(last_start + 72)]
