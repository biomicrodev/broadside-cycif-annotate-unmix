import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from natsort import natsorted
from ome_types import from_xml, OME
from ome_types.model import Pixels
from ome_zarr.io import parse_url
from pint import Quantity
from tifffile import TiffReader

from broadside.utils import get_high_contrast_color, get_multiplex_color
from broadside.utils.geoms import Point2D
from broadside.utils.units import ureg, default_distance_units

os.environ["NAPARI_ASYNC"] = "1"
os.environ["NAPARI_OCTREE"] = "1"
from napari.layers import Image

re_round = re.compile("^R.+$", flags=re.IGNORECASE)
tile_filename_glob = "*.ome.tiff"


@dataclass(frozen=True)
class Channel:
    index: int
    image_layer: Image
    pyramid: list[da.Array]
    biomarker: str
    fluorophore: str


Rounds = dict[str, list[Channel]]


def _get_rounds_from_fs(scene_dir: Path) -> set[str]:
    tiles_dir = scene_dir / "tiles"

    names = set()
    for p in tiles_dir.iterdir():
        is_round_name = p.is_dir() and (re_round.match(p.stem))
        if not is_round_name:
            continue
        names.add(p.stem)

    return names


def _get_pixel_size(pixels: Pixels) -> Quantity:
    pixel_size_x: Quantity = pixels.physical_size_x * getattr(
        ureg, pixels.physical_size_x_unit.name.lower()
    )
    pixel_size_y: Quantity = pixels.physical_size_y * getattr(
        ureg, pixels.physical_size_y_unit.name.lower()
    )
    if not np.allclose(pixel_size_x, pixel_size_y):
        warnings.warn(f"Non-square pixel size not supported")
    pixel_size = (pixel_size_x / 2) + (pixel_size_y / 2)
    return pixel_size


def _get_position(pixels: Pixels):
    first_plane = pixels.planes[0]
    x = first_plane.position_x * getattr(ureg, first_plane.position_x_unit.name.lower())
    y = first_plane.position_y * getattr(ureg, first_plane.position_y_unit.name.lower())
    return Point2D(x=x, y=y)


class Scene:
    def __init__(self, path: Path, *, specific_rounds: set[str] | None = None):
        self._dir = path

        self.log = logging.getLogger(f"Scene({self.name})")

        round_names = _get_rounds_from_fs(path)
        if specific_rounds is not None:
            round_names &= specific_rounds
        self._round_names = natsorted(list(round_names))

        self._rounds: dict[str, list[Channel]] | None = None
        self._pyramid: list[da.Array] | None = None
        self._pixel_size: Quantity | None = None

    @property
    def name(self) -> str:
        return self._dir.stem

    @property
    def round_names(self) -> list[str]:
        return self._round_names

    def tile_paths(self, round_name: str) -> list[Path]:
        tiles_dir = self._dir / "tiles" / round_name
        return natsorted(list(tiles_dir.glob(tile_filename_glob)), key=lambda p: p.stem)

    @property
    def ome_tiff_path(self):
        return self._dir / "image.ome.tiff"

    @property
    def ome_zarr_path(self):
        return self._dir / "image.ome.zarr"

    @property
    def ome_xml_path(self):
        return self._dir / "image.ome.xml"

    @lru_cache()
    def timestamp(self, round_name: str) -> datetime:
        """
        from_xml is VERY slow, the very first call to from_xml takes nearly a second;
        we parse the xml directly for this step, but because we value correctness at
        this point in time, we'll parse the entire ome-xml using ome-types
        """
        tile_path = self.tile_paths(round_name)[0]

        with TiffReader(tile_path) as reader:
            ome = from_xml(reader.ome_metadata, parser="lxml")

        n_images = len(ome.images)
        if n_images != 1:
            warnings.warn(
                f"Expected one image, found {n_images}; using the first image"
            )

        return ome.images[0].acquisition_date

    def load_pyramid(self):
        with self.ome_xml_path.open("r") as file:
            ome: OME = from_xml(file.read(), parser="lxml")
            pixels: Pixels = ome.images[0].pixels
            channels = pixels.channels
            max_val = 2**pixels.significant_bits
            pixel_size = _get_pixel_size(pixels)
            position = _get_position(pixels)

        # compute scales and translations
        scale = pixel_size.to(default_distance_units).magnitude
        root = zarr.open_group(store=parse_url(self.ome_zarr_path, mode="r").store)
        pyramid = [da.from_zarr(root.store, key) for key in root.keys()]

        # get properties
        center = Point2D(
            x=(position.x.to(default_distance_units)).magnitude,
            y=(position.y.to(default_distance_units)).magnitude,
        )
        half_shape = Point2D(
            y=pyramid[0][0].shape[0] * scale / 2,
            x=pyramid[0][0].shape[1] * scale / 2,
        )
        translate = center - half_shape

        # need to refactor this when we actually apply the unmixing matrix
        # organize dask arrays into rounds and channels
        rounds: Rounds = dict()

        for i, channel in enumerate(channels):
            channel_params = json.loads(channel.name)
            # keys: cycle, channel, biomarker, fluorophore
            channel_data = [d[i] for d in pyramid]

            # get most downsampled level of the pyramid for fast percentile checking
            lo, hi = da.percentile(channel_data[-1].ravel(), (30, 99.99)).compute()
            lo = max(lo, 0)
            hi = min(max(1, hi), max_val)

            round_name = channel_params["cycle"]
            channel_name = f'{self.name}: {channel_params["biomarker"]}'

            image = Image(
                channel_data,
                name=channel_name,
                contrast_limits=[lo, hi],
                blending="additive",
                multiscale=True,
                scale=(scale, scale),
                translate=(translate.y, translate.x),
                visible=False,
                # cache=True,
            )
            # try to prevent thumbnail from being updated
            # the action itself is fast, but it adds up
            image.events.thumbnail.block()
            image._slice.loaded = False
            image.contrast_limits_range = [0, max_val]

            channel = Channel(
                image_layer=image,
                index=channel_params["channel"],
                biomarker=channel_params["biomarker"],
                fluorophore=channel_params["fluorophore"],
                pyramid=channel_data,
            )

            try:
                rounds[round_name].append(channel)
            except KeyError:
                rounds[round_name] = [channel]

        for channels in rounds.values():
            for i, channel in enumerate(channels):
                # show first channel on initial load
                if i == 0:
                    channel.image_layer.visible = True

        for channels in rounds.values():
            for channel in channels:
                channel.image_layer._slice.loaded = True
                channel.image_layer.events.thumbnail.unblock()
                channel.image_layer._update_thumbnail()

        self._max_val = max_val
        self._rounds = rounds
        self._pyramid = pyramid
        self._translate = translate
        self._scale = scale

    @property
    def max_val(self):
        return self._max_val

    @property
    def translate(self):
        return self._translate

    @property
    def scale(self):
        return self._scale

    @property
    def rounds(self):
        return self._rounds

    @property
    def pyramid(self):
        return self._pyramid

    @property
    def bbox(self):
        base_level = 0
        ref_channel = 0
        shape = self.pyramid[base_level][ref_channel].shape
        points_px = [
            Point2D(y=0, x=0),
            Point2D(y=shape[0], x=0),
            Point2D(y=shape[0], x=shape[1]),
            Point2D(y=0, x=shape[1]),
        ]
        points_w = [
            Point2D(
                x=p.x * self._scale + (self._translate.x),
                y=p.y * self._scale + (self._translate.y),
            )
            for p in points_px
        ]
        return points_w

    def apply_high_contrast_cmap(self):
        for channels in self.rounds.values():
            for i, channel in enumerate(channels):
                channel.image_layer.colormap = get_high_contrast_color(i)

    def apply_multiplex_cmap(self):
        for channels in self.rounds.values():
            for i, channel in enumerate(channels):
                channel.image_layer.colormap = get_multiplex_color(i)
