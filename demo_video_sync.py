import os

import bab_datasets as nod


def main():
    video_dir = os.getenv("BAB_DATASETS_VIDEO_DIR")
    if not video_dir:
        video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos_BAB"))

    sync = nod.sync_video_with_dataset(
        video_name="swept_sine",
        dataset_name="swept_sine",
        video_dir=video_dir,
        roi="lower_left",
        roi_frac=0.25,
    )

    print("Video:", sync.video_name)
    print("Dataset:", sync.dataset_name)
    print("FPS:", sync.fps)
    print("Sampling time:", sync.sampling_time)
    print("Video frame start:", sync.frame_start)
    print("Dataset sample start:", sync.sample_start)


if __name__ == "__main__":
    main()
