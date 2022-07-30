from pathlib import Path
from typing import List
from core.image_processing.png_proc import DflPng
from client.seged import seged

dst_step3_path = Path("/home/kay/Desktop/xseg/data_dst/3_xseg_editor/")
src_step3_path = Path("/home/kay/Desktop/xseg/data_src/3_xseg_editor/")
dst_step4_path = Path("/home/kay/Desktop/xseg/data_dst/4_1_xseg_training/")
src_step4_path = Path("/home/kay/Desktop/xseg/data_src/4_1_xseg_training/")
xseg_model_path = Path("/home/kay/Desktop/xseg/data_src/4_2_xseg_model/")


def first_time_setup_extracted_fs_faces_for_xseg(step_3_image_dir: Path):
    images: List[Path] = list(x for x in step_3_image_dir.iterdir())
    for image in images:
        dfl_png_img: DflPng = DflPng.load(image)
        dfl_png_img.save()


def launch_xseg_editor(aligned_data_path: Path):
    seged.start_segmentation_editor(aligned_data_path)


if __name__ == "__main__":
    # first_time_setup_extracted_fs_faces_for_xseg(src_step3_path)
    launch_xseg_editor(src_step3_path)
