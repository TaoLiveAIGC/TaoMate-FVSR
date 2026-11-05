# Automatic face cropping

The `yolo_face` mode uses CPU Ultralytics and a local `face_yolov8n.pt`. It does not download weights while executing a workflow. Frames are normalized to 25 fps before detection and read individually from disk.

The tracker first selects the largest face, then associates boxes by overlap. Detection boxes are expanded upward by 20%, made square with a 1.45 expansion, processed with the pinned upstream adaptive box helper, and scaled around their center. `crop_scale=0.8` is the default starting point. Inspect the preview and adjust for your input. This is a crop-side multiplier, not model scale2 or whole-video upscaling; the geometry is not identical to DWPose landmark cropping.

The workflow targets one person in one continuous shot. Geometric association is not identity recognition. Missing detections retain their frame index and audio timing; input crops use the nearest valid geometry on the current frame, while paste-back leaves invalid frames unchanged. All-missing detection raises an error. Suspected cuts are rejected so that a reference image and model history cannot silently cross shots. Flashes and fast motion can trigger this heuristic; split the clip into shots before processing.

The crop helper returns `boxes` (integer original-frame xyxy coordinates), `valid` flags and cut indices. Invalid boxes must not be pasted. Track data is stored in the temporary `crop-track.npz`. Multi-person crossings, tiny faces and fast profiles need visual inspection.

Unless explicitly set by the caller, `YOLO_CONFIG_DIR` points to a process temporary directory before Ultralytics import. Node imports do not initialize CUDA.
