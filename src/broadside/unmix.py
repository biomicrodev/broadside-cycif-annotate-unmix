import math
import sys
from dataclasses import dataclass
from typing import Iterator

import dask.array as da
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from picasso.unmixing import minimize_mi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_erosion, disk
from skimage.transform import rescale
from skimage.util import img_as_float
from tqdm import tqdm

from broadside.scenes import Scene

rng = default_rng(seed=0)


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and (n != 0)


def get_image_for_unmixing(
    *,
    scenes: list[Scene],
    round_name: str,
    n_pixels: int,
    ref_channel=0,
    channels: list[int] = None,
    downscale_for_tissue_seg=128,
    downscale_for_unmix=8,
) -> npt.NDArray:
    """
    TODO: refactor n_pixels to be more meaningful

    for each scene:
    1. find parts of scene that have tissue
        1. get ref channel of top layer of pyramid and obtain image at
            downscale_for_tissue_seg level
        2. threshold
        3. erode
    2. pick n points at downscale_for_tissue_seg level, proportionately to the tissue
        area
    3. get slices of pyramid at downscale_for_unmix level, for the desired channels in the round
    4. stack, and threshold based on downscale
    """
    # validate powers of two
    assert _is_power_of_two(downscale_for_tissue_seg)
    assert _is_power_of_two(downscale_for_unmix)
    assert len(scenes) >= 1
    assert downscale_for_tissue_seg != downscale_for_unmix

    top_layer_inds_by_scene = []

    base_layer = 0
    top_layer = -1

    # we would like to unmix using parts of the slide that have tissue
    for scene in scenes:
        # get reference channel of desired round of top layer
        top_layer_ref_channel: npt.NDArray = (
            scene.rounds[round_name][ref_channel].pyramid[top_layer].compute()
        )
        top_layer_ref_channel = img_as_float(top_layer_ref_channel)

        downscale_between_base_and_top_layers = int(
            round(
                scene.pyramid[base_layer][ref_channel].shape[0]
                / scene.pyramid[top_layer][ref_channel].shape[0]
            )
        )
        # print("ds pyramid", downscale_between_base_and_top_layers)
        downscale_remaining = (
            downscale_for_tissue_seg / downscale_between_base_and_top_layers
        )
        if downscale_remaining > 1:
            # print(f"top layer ref im rescaled by {downscale_remaining}")
            top_layer_ref_channel = rescale(
                top_layer_ref_channel, 1.0 / downscale_remaining
            )

        top_layer_ref_channel = gaussian(top_layer_ref_channel)
        threshold = threshold_otsu(top_layer_ref_channel)
        top_layer_ref_binary = top_layer_ref_channel >= threshold
        top_layer_ref_binary = binary_erosion(top_layer_ref_binary, footprint=disk(2))
        top_layer_inds = top_layer_ref_binary.nonzero()

        top_layer_inds_by_scene.append(top_layer_inds)

    if channels is None:
        channels = list(range(len(scenes[0].rounds[round_name])))

    n_inds_total = 0
    for inds_y, _ in top_layer_inds_by_scene:
        n_inds_total += len(inds_y)

    unmix_chunks = []
    for (inds_y, inds_x), scene in zip(top_layer_inds_by_scene, scenes):
        n_inds_in_scene = int(round(n_pixels * len(inds_y) / n_inds_total))
        # print(f"n inds for {scene.name}", n_inds_in_scene)

        possible_inds = np.arange(len(inds_y))
        chosen_inds_in_scene = rng.choice(possible_inds, n_inds_in_scene)

        unmix_layer = int(round(math.log2(downscale_for_unmix)))
        downscale_difference = downscale_for_tissue_seg / downscale_for_unmix

        # convert indexes at ds_for_tissue_seg to ds_unmix level
        iys = inds_y[chosen_inds_in_scene]
        ixs = inds_x[chosen_inds_in_scene]
        for iy, ix in zip(iys, ixs):
            chunk_by_channels = [
                scene.rounds[round_name][channel].pyramid[unmix_layer][
                    iy * downscale_difference : (iy + 1) * downscale_difference,
                    ix * downscale_difference : (ix + 1) * downscale_difference,
                ]
                for channel in channels
            ]
            chunk = da.stack(chunk_by_channels)
            unmix_chunks.append(chunk)

    image_for_unmixing = da.concatenate(unmix_chunks, axis=1).compute()
    assert image_for_unmixing.ndim == 3

    image_for_unmixing = img_as_float(image_for_unmixing)
    saturation_thresh_for_unmix = (1.0 / downscale_for_unmix) * 0.5
    image_for_unmixing[image_for_unmixing > saturation_thresh_for_unmix] = 0.0
    return image_for_unmixing


@dataclass(frozen=True)
class UnmixProgress:
    matrix: npt.NDArray
    loss: npt.NDArray


def compute_unmixing_matrix_as_gen(
    image: npt.NDArray, *, max_iters=1_000, step_mult=0.1, verbose=False
) -> Iterator[UnmixProgress]:
    # validate inputs
    assert image.ndim == 3  # CYX

    n_channels = image.shape[0]

    image = img_as_float(image)
    image_orig = image.copy()

    mat_inc = np.eye(n_channels, dtype=float)
    mat_inc_last = np.eye(n_channels, dtype=float)

    mat_cumul = np.eye(n_channels, dtype=float)
    mat_cumul_last = np.eye(n_channels, dtype=float)

    for _ in tqdm(
        range(max_iters),
        disable=not verbose,
        desc="Unmixing iters",
        total=0,
        file=sys.stdout,
    ):
        # compute incremental change in unmixing matrix
        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    mat_inc[row, col] = 1.0
                else:
                    coef = minimize_mi(
                        image[col], image[row], init_alpha=mat_inc_last[row, col]
                    )
                    mat_inc[row, col] = -step_mult * coef
        mat_inc_last = mat_inc

        # update cumulative matrix based on incremental matrix
        mat_cumul = mat_inc @ mat_cumul

        # constrain coefficients
        for row in range(n_channels):
            for col in range(n_channels):
                if row == col:
                    mat_cumul[row, col] = 1.0
                else:
                    if mat_cumul[row, col] > 0.0:
                        mat_cumul[row, col] = 0.0

        # yield progress
        mat_cumul_mad = np.sum(np.abs(mat_cumul_last - mat_cumul))
        yield UnmixProgress(matrix=mat_cumul.copy(), loss=mat_cumul_mad)

        # check stop condition
        if np.allclose(mat_cumul_last, mat_cumul):
            break
        mat_cumul_last = mat_cumul

        # update image
        image = np.tensordot(mat_cumul, image_orig, axes=1)
