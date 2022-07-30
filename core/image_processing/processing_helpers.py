import os
import struct
from ast import literal_eval
from zlib import crc32
import cv2
import numpy as np
import numpy.linalg as npla


def read_image(filename, with_metadata=False):
    """ Read an image file from a file location.

    Extends the functionality of :func:`cv2.imread()` by ensuring that an image was actually
    loaded. Errors can be logged and ignored so that the process can continue on an image load
    failure.

    Parameters
    ----------
    filename: str
        Full path to the image to be loaded.
    raise_error: bool, optional
        If ``True`` then any failures (including the returned image being ``None``) will be
        raised. If ``False`` then an error message will be logged, but the error will not be
        raised. Default: ``False``
    with_metadata: bool, optional
        Only returns a value if the images loaded are extracted Faceswap faces. If ``True`` then
        returns the Faceswap metadata stored with in a Face images .png_meta exif header.
        Default: ``False``

    Returns
    -------
    numpy.ndarray or tuple
        If :attr:`with_metadata` is ``False`` then returns a `numpy.ndarray` of the image in `BGR`
        channel order. If :attr:`with_metadata` is ``True`` then returns a `tuple` of
        (`numpy.ndarray`" of the image in `BGR`, `dict` of face's Faceswap metadata)
    Example
    -------
    >>> image_file = "/path/to/image.png_meta"
    >>> try:
    >>>    image = read_image(image_file, raise_error=True, with_metadata=False)
    >>> except:
    >>>     raise ValueError("There was an error")
    """
    filename = str(filename)
    if not with_metadata:
        retval = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
        if retval is None:
            raise ValueError("Image is None")
    else:
        with open(filename, "rb") as infile:
            raw_file = infile.read()
            metadata = png_read_meta(raw_file)
        image = cv2.imdecode(np.frombuffer(raw_file, dtype="uint8"), cv2.IMREAD_UNCHANGED)
        retval = (image, metadata)
    return retval


def png_read_meta(itxt_chunk):
    """ Read the Faceswap information stored in a png_meta's iTXt field.

    Parameters
    ----------
    itxt_chunk: bytes
        The bytes encoded png_meta file to read header data from

    Returns
    -------
    dict
        The Faceswap information stored in the png_meta header

    Notes
    -----
    This is a very stripped down, non-robust and non-secure header reader to fit a very specific
    task. OpenCV will not write any iTXt headers to the png_meta file, so we make the assumption that
    the only iTXt header that exists is the one that Faceswap created for storing alignments.
    """
    retval = None
    pointer = 0

    while True:
        pointer = itxt_chunk.find(b"iTXt", pointer) - 4
        if pointer < 0:
            print("No metadata in png_meta")
            break
        length = struct.unpack(">I", itxt_chunk[pointer:pointer + 4])[0]
        pointer += 8
        keyword, value = itxt_chunk[pointer:pointer + length].split(b"\0", 1)
        if keyword == b"faceswap":
            retval = literal_eval(value[4:].decode("utf-8"))
            break
        print("Skipping iTXt chunk: '%s'", keyword.decode("latin-1", "ignore"))
        pointer += length + 4
    return retval


def pack_to_itxt(metadata):
    """ Pack the given metadata dictionary to a PNG iTXt header field.

    Parameters
    ----------
    metadata: dict or bytes
        The dictionary to write to the header. Can be pre-encoded as utf-8.

    Returns
    -------
    bytes
        A byte encoded PNG iTXt field, including chunk header and CRC
    """
    if not isinstance(metadata, bytes):
        metadata = str(metadata).encode("utf-8", "strict")
    key = "faceswap".encode("latin-1", "strict")

    chunk = key + b"\0\0\0\0\0" + metadata
    crc = struct.pack(">I", crc32(chunk, crc32(b"iTXt")) & 0xFFFFFFFF)
    length = struct.pack(">I", len(chunk))
    retval = length + b"iTXt" + chunk + crc
    return retval


def update_existing_metadata(filename, metadata):
    """ Update the png header metadata for an existing .png extracted face file on the filesystem.

    Parameters
    ----------
    filename: str
        The full path to the face to be updated
    metadata: dict or bytes
        The dictionary to write to the header. Can be pre-encoded as utf-8.
    """

    tmp_filename = filename + "~"
    with open(filename, "rb") as png, open(tmp_filename, "wb") as tmp:
        chunk = png.read(8)
        if chunk != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid header found in png: {filename}")
        tmp.write(chunk)

        while True:
            chunk = png.read(8)
            length, field = struct.unpack(">I4s", chunk)

            if field == b"IDAT":  # Write out all remaining data
                tmp.write(chunk + png.read())
                break

            if field != b"iTXt":  # Write non iTXt chunk straight out
                tmp.write(chunk + png.read(length + 4))  # Header + CRC
                continue

            keyword, value = png.read(length).split(b"\0", 1)
            if keyword != b"faceswap":
                # Write existing non fs-iTXt data + CRC
                tmp.write(keyword + b"\0" + value + png.read(4))
                continue

            tmp.write(pack_to_itxt(metadata))
            png.seek(4, 1)  # Skip old CRC

    os.replace(tmp_filename, filename)


def dist_to_edges(pts, pt, is_closed=False):
    """
    returns array of dist from pt to edge and projection pt to edges
    """
    if is_closed:
        a = pts
        b = np.concatenate((pts[1:, :], pts[0:1, :]), axis=0)
    else:
        a = pts[:-1, :]
        b = pts[1:, :]

    pa = pt - a
    ba = b - a

    div = np.einsum('ij,ij->i', ba, ba)
    div[div == 0] = 1
    h = np.clip(np.einsum('ij,ij->i', pa, ba) / div, 0, 1)

    x = npla.norm(pa - ba * h[..., None], axis=1)

    return x, a + ba * h[..., None]


def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter + fmt_size,) + struct.unpack(fmt, data[counter:counter + fmt_size])


def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[..., np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat(img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[..., 0:target_channels]
        c = target_channels

    return img
