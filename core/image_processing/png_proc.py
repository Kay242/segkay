import copy
import pickle
from pathlib import Path
from typing import Optional, Tuple, Any
from zlib import decompress
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from core.image_processing.aligned_face import AlignedFace
from core.image_processing.seg_polygons import SegIEPolys
from core.image_processing.processing_helpers import read_image, normalize_channels, update_existing_metadata
from core.image_processing.PNGMETA import FaceswapMetaKeys, DeepFaceLabMetaKeys, FaceswapMasks, FACESWAP_MASKS,\
    META_KEYS_TO_ENCODE


class DflPng:
    def __init__(self, file_path: Path):
        self.filename = file_path
        self.metadata: dict = {}
        self._decoded_image = None
        self._shape = None
        self._face_type = "whole_face"

    @staticmethod
    def load(filename: Path):
        """
        This function should be called first and is the primary method of instantiating a DflPng class. This will load
        the provided image, retrieve all the data and metadata and arrange it in a usable method.

        Parameters
        ----------
        filename : Path
            Filepath to image as a Path object.
        Returns
        -------
        dflpng : DflPng
            Instantiated DflPng object
        """
        dflpng: DflPng = DflPng(filename)
        dflpng._decoded_image, dflpng.metadata = read_image(str(filename), with_metadata=True)

        if dflpng.metadata:
            if DeepFaceLabMetaKeys.extracted_landmarks not in dflpng.metadata:
                fs_landmarks_meta = dflpng.metadata[FaceswapMetaKeys.alignments][FaceswapMetaKeys.landmarks]
                extracted_landmarks = np.asarray(fs_landmarks_meta.copy())
                source_image_dimensions: tuple = dflpng.metadata[FaceswapMetaKeys.source][FaceswapMetaKeys.frame_dims]
                extracted_landmarks[:, 0] *= dflpng.decoded_image.shape[1] / source_image_dimensions[1]
                extracted_landmarks[:, 1] *= dflpng.decoded_image.shape[0] / source_image_dimensions[0]
                dflpng.metadata[DeepFaceLabMetaKeys.extracted_landmarks] = extracted_landmarks

            for loaded_key in dflpng.metadata.keys():
                # Some objects needed to be saved as bytes objects and must be loaded via pickle
                if loaded_key in META_KEYS_TO_ENCODE:
                    loaded_value = dflpng.metadata[loaded_key]
                    if type(loaded_value) is bytes:
                        dflpng.metadata[loaded_key] = pickle.loads(loaded_value)
        return dflpng

    @property
    def source_shape(self) -> Tuple[int, int]:
        """

        Returns
        -------
        tuple
            tuple of the shape of the source image as (height, width)
        """
        source_height, source_width = self.metadata[FaceswapMetaKeys.source][FaceswapMetaKeys.frame_dims]
        return source_height, source_width

    @property
    def face_centered_image(self):
        al_face = AlignedFace(self.source_landmarks,
                              self.decoded_image,
                              size=self.shape[0],
                              centering="face",
                              is_aligned=True)
        return al_face.face

    @property
    def center_face_masked(self, masker: FaceswapMasks = FaceswapMasks.bisenet_fp_face, debug=True):
        stored_mask, _ = self.faceswap_mask(masker, return_mask_parent_dict=True)
        extracted_image_shape = self.shape[:2]
        face = self.face_centered_image

        extracted_frame_mask = cv2.resize(stored_mask, extracted_image_shape, interpolation=cv2.INTER_AREA)
        masked_face = np.concatenate((face, np.expand_dims(extracted_frame_mask, axis=-1)), axis=-1)

        if debug:
            f, sub_plot = plt.subplots(1, 2)
            sub_plot[0].imshow(face)
            sub_plot[1].imshow(masked_face)
            plt.show()

        return masked_face

    @staticmethod
    def zoomed_affinity(affine_matrix,
                        mask_storage_size,
                        face_pixel_edge_size,
                        inverse=False) -> np.ndarray:
        if inverse:
            zoom = mask_storage_size / face_pixel_edge_size
        else:
            zoom = face_pixel_edge_size / mask_storage_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])

        zoomed_affine_matrix = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))

        return zoomed_affine_matrix

    def _retrieve_pickled_transformation(self,
                                         source_image: np.ndarray,
                                         face_pixel_edge_size: int = 128,
                                         masker: FaceswapMasks = FaceswapMasks.bisenet_fp_face):
        """
        FOR DEBUGGING ONLY.
        Parameters
        ----------
        source_image
        face_pixel_edge_size
        masker

        Returns
        -------

        """
        stored_mask, mask_meta_dict = self.faceswap_mask(masker, return_mask_parent_dict=True)
        mask_stored_size = mask_meta_dict["stored_size"]
        stored_affine_matrix = np.asarray(mask_meta_dict["affine_matrix"])

        source_height, source_width = self.source_shape
        extracted_image_shape = self.shape[:2]

        resized_face_mask = cv2.resize(stored_mask, extracted_image_shape, interpolation=cv2.INTER_AREA)

        adjusted_affine_matrix = self.zoomed_affinity(stored_affine_matrix,
                                                      mask_stored_size,
                                                      extracted_image_shape[0],
                                                      inverse=False)

        self.set_affine_matrix(adjusted_affine_matrix)
        self.save()

        # cropped to extracted face zone
        transformed_image = cv2.warpAffine(source_image[..., 2::-1],
                                           adjusted_affine_matrix,
                                           extracted_image_shape,
                                           flags=cv2.INTER_AREA,
                                           borderMode=cv2.BORDER_CONSTANT)
        masked_extracted_face = np.concatenate((transformed_image, resized_face_mask[..., None]), axis=-1)

        dst_frame = np.zeros((source_height, source_width, 3), dtype="uint8")
        adjusted_affine_matrix2 = self.zoomed_affinity(stored_affine_matrix,
                                                       mask_stored_size,
                                                       face_pixel_edge_size,
                                                       inverse=False)
        # cv2.wrapAffine needs shape given as (width, height) to return an array with shape (height, width). Basically
        # it needs the transposed shape.
        mask: np.ndarray = cv2.warpAffine(stored_mask,
                                          adjusted_affine_matrix2,
                                          (source_width, source_height),
                                          dst_frame,
                                          flags=cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT)

        masked_face_original_source_frame = np.concatenate((source_image, np.expand_dims(mask, axis=-1)), axis=-1)
        masked_face_empty_source_frame = np.concatenate((dst_frame, np.expand_dims(mask, axis=-1)), axis=-1)

        f, sub_plot = plt.subplots(2, 2)
        sub_plot[0, 0].imshow(transformed_image)
        sub_plot[0, 1].imshow(masked_extracted_face)
        sub_plot[1, 0].imshow(masked_face_original_source_frame)
        sub_plot[1, 1].imshow(masked_face_empty_source_frame)
        plt.show()

    def save(self) -> None:
        # Initially assign object by reference in case nothing needs to be done
        metadata_to_save = self.metadata.copy()

        for loaded_key in self.metadata.keys():
            if loaded_key in META_KEYS_TO_ENCODE:
                loaded_value = self.metadata[loaded_key]
                # At this point the data SHOULD NOT be in bytes form.
                if type(loaded_value) is not bytes:
                    # xseg polys need to be saved a bytes object to avoid issues with ast.literal_eval
                    # later when the meta_data is re-read by the png_read_meta function.
                    if loaded_key in [DeepFaceLabMetaKeys.seg_ie_polys, DeepFaceLabMetaKeys.xseg_mask]:
                        # regular copy top level dict entries so changes to top level entries are not reflected in
                        # the original object (self.metadata). However this strictly applies to the top level,
                        # for example, if one top level element in a dict is a list value and an element in that list
                        # is changed in the copy it will still be reflected in the original. Whereas if you replace
                        # that list entirely with another list or something else it won't be reflected in the
                        # original. Since we are only interested in replacing the seg_polys value with a bytes
                        # representation without affecting the original. A deepcopy is needed (but just for that
                        # element in the dict).
                        metadata_to_save[loaded_key] = copy.deepcopy(loaded_value)
                        metadata_to_save[loaded_key] = pickle.dumps(loaded_value)
                    else:
                        metadata_to_save[loaded_key] = pickle.dumps(loaded_value)

        update_existing_metadata(str(self.filename), metadata_to_save)

    @property
    def has_metadata(self):
        return len(self.metadata.keys()) != 0

    @property
    def decoded_image(self) -> np.ndarray:
        if self._decoded_image is None:
            self._decoded_image = cv2.imread(str(self.filename), flags=cv2.IMREAD_UNCHANGED)
        return self._decoded_image

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self._shape = self.decoded_image.shape
        return self._shape

    @property
    def face_type(self) -> str:
        return self._face_type

    def set_face_type(self, face_type) -> None:
        self._face_type = face_type

    @property
    def extracted_landmarks(self) -> np.ndarray:
        extracted_landmarks = self.metadata.get(DeepFaceLabMetaKeys.extracted_landmarks, None)
        return extracted_landmarks

    def set_extracted_landmarks(self, extracted_landmarks: np.ndarray) -> None:
        """
        Set the 68 point-based landmarks for the extracted image.
        @param extracted_landmarks: numpy array of the landmark coordinates in the extracted image.
        @return: None.
        """
        self.metadata[DeepFaceLabMetaKeys.extracted_landmarks] = extracted_landmarks

    @property
    def source_filename(self) -> str:
        faceswap_source_dict = self.metadata.get(FaceswapMetaKeys.source, None)
        source_file_name = faceswap_source_dict.get(FaceswapMetaKeys.source_filename, None)
        return source_file_name

    def set_source_filename(self, source_filename: str) -> None:
        faceswap_source_dict = self.metadata[FaceswapMetaKeys.source]
        faceswap_source_dict[FaceswapMetaKeys.source_filename] = source_filename

    @property
    def source_rect(self) -> np.ndarray:
        return self.metadata.get(DeepFaceLabMetaKeys.source_rect, None)

    def set_source_rect(self, source_rect: np.ndarray) -> None:
        self.metadata[DeepFaceLabMetaKeys.source_rect] = source_rect

    @property
    def source_landmarks(self) -> np.ndarray:
        faceswap_dict: dict = self.metadata.get(FaceswapMetaKeys.alignments, None)
        source_landmarks = faceswap_dict.get(FaceswapMetaKeys.landmarks, None)
        return np.array(source_landmarks) if source_landmarks else None

    def set_source_landmarks(self, source_landmarks: np.ndarray) -> None:
        """

        @param source_landmarks: A numpy array of the 68 point-based landmarks for the source image.
        @return: None.
        """
        faceswap_dict: dict = self.metadata[FaceswapMetaKeys.alignments]
        faceswap_dict[FaceswapMetaKeys.landmarks] = source_landmarks

    @property
    def affine_matrix(self) -> np.ndarray:
        mat = self.metadata.get(DeepFaceLabMetaKeys.image_to_face_mat, None)
        return mat

    def set_affine_matrix(self, affine_matrix: np.ndarray) -> None:
        self.metadata[DeepFaceLabMetaKeys.image_to_face_mat] = affine_matrix

    @property
    def has_xpolys(self) -> bool:
        return self.metadata.get(DeepFaceLabMetaKeys.seg_ie_polys) is not None

    @property
    def xpolys(self) -> SegIEPolys:
        """

        @return: A SegIEPolys class.
        """
        dict_val = self.metadata.get(DeepFaceLabMetaKeys.seg_ie_polys, None)
        if dict_val is not None:
            d = SegIEPolys.load(dict_val)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys: SegIEPolys) -> None:
        """
        Add polys to image metadata.

        @param seg_ie_polys: Object of type SegIEPolys.
        @return: None
        """
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.metadata[DeepFaceLabMetaKeys.seg_ie_polys] = seg_ie_polys

    @property
    def has_xmask(self):
        return self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None) is not None

    @property
    def compressed_xmask(self) -> Optional[ndarray]:
        """

        @return: numpy array of an Xseg compressed mask or None if no mask is present.
        """
        mask_buf = self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None)
        if mask_buf is None:
            return None

        return mask_buf

    def faceswap_mask(self, mask_name: FaceswapMasks, return_mask_parent_dict=False) -> Tuple[Any, Any]:
        """

        Parameters
        ----------
        mask_name
        return_mask_parent_dict

        Returns
        -------

        """
        try:
            masks_in_metadata = self.metadata[FaceswapMetaKeys.alignments][FaceswapMetaKeys.alignments_mask]
        except KeyError as e:
            raise Exception("Missing faceswap alignments/masks in metadata. Double check that the image {0} has been"
                            " correctly extracted from faceswap.".format(self.filename.name)) from e

        if mask_name not in FACESWAP_MASKS:
            raise KeyError("Provided mask name, {0}, is either incorrect or not currently supported.".format(mask_name))
        try:
            mask_parent_dict = masks_in_metadata[mask_name]
        except KeyError:
            raise KeyError("Provided mask name, {0}, is missing from metadata".format(mask_name))
        mask_compressed_array = mask_parent_dict["mask"]
        stored_size = mask_parent_dict["stored_size"]
        dims = (stored_size, stored_size, 1)

        decompressed_mask = np.frombuffer(decompress(mask_compressed_array), dtype="uint8").reshape(dims)
        if return_mask_parent_dict:
            return decompressed_mask, mask_parent_dict
        else:
            return decompressed_mask

    @property
    def xmask(self) -> Optional[ndarray]:
        """

        Returns
        -------
        ndarray
            Returns an array of the decompressed Xseg mask or None if no mask is available.
        """
        mask_buf = self.metadata.get(DeepFaceLabMetaKeys.xseg_mask, None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 2:
            img = img[..., None]

        return np.asarray(img.astype(np.float32) / 255.0)

    def set_xmask(self, mask_a: np.ndarray) -> None:
        if mask_a is None:
            self.metadata[DeepFaceLabMetaKeys.xseg_mask] = None
            return

        mask_a = normalize_channels(mask_a, 1)
        img_data = np.clip(mask_a * 255, 0, 255).astype(np.uint8)

        ret, buf = cv2.imencode('.png', img_data)

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.metadata[DeepFaceLabMetaKeys.xseg_mask] = buf

    def convert_xmask_to_fs_mask(self, mask_storage_size=128):
        """

        @param mask_storage_size: The size that the mask array is stored as, default 128.
        @return: A faceswap readable and understandable mask in BGR format.
        """
        xseg_mask = self.xmask
        if not self.xmask.any():
            return None

        mask = cv2.resize(xseg_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1)
        fs_mask_from_xseg = (1 - normalized_mask) * 0.5 + normalized_mask
        fs_mask_from_xseg_clipped = np.clip(fs_mask_from_xseg * 255, 0, 255).astype(np.uint8)
        fs_stored_mask = cv2.resize(fs_mask_from_xseg_clipped, (mask_storage_size, mask_storage_size),
                                    interpolation=cv2.INTER_AREA)

        return fs_stored_mask

    def xmask_image_overlay(self):
        """

        @return: Extracted face overlain by Xseg generated mask in BGR channel format.
        """
        xseg_mask = self.xmask
        if not self.xmask.any():
            return None

        mask = cv2.resize(xseg_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1)
        xseg_image = self.decoded_image.astype(np.float32) / 255.0
        xseg_overlay_mask = xseg_image * (1 - normalized_mask) * 0.5 + xseg_image * normalized_mask
        clipped_xseg_overlay_mask = np.clip(xseg_overlay_mask * 255, 0, 255).astype(np.uint8)

        return clipped_xseg_overlay_mask

    def debug_fs_bisenet_face_overlay(self):
        """
        Returns the extracted face image array overlain by the bisenet-fp_face mask for debugging purposes.
        @return:
        """

        bisenet_face_mask = self.faceswap_mask("bisenet-fp_face", return_mask_parent_dict=False)

        mask = cv2.resize(bisenet_face_mask, self.shape[:2], interpolation=cv2.INTER_AREA)
        normalized_mask = normalize_channels(mask, 1) / 255.
        image = self.decoded_image.astype(np.float32) / 255.
        bisenet_mask_overlay = image * (1 - normalized_mask) * 0.5 + image * normalized_mask
        clipped_bisenet_mask_overlay = np.clip(bisenet_mask_overlay * 255, 0, 255).astype(np.uint8)

        fs_mask_from_xseg = (1 - normalized_mask) * 0.5 + normalized_mask
        fs_mask_from_xseg_clipped = np.clip(fs_mask_from_xseg * 255, 0, 255).astype(np.uint8)
        fs_stored_mask = cv2.resize(fs_mask_from_xseg_clipped, (128, 128), interpolation=cv2.INTER_AREA)

        return mask, clipped_bisenet_mask_overlay, fs_stored_mask
