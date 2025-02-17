import argparse
from typing import List

import torch

from lightglue_pytorch_with_plugin import LightGlue, SuperPoint, SuperPointGridSampler
from lightglue_pytorch_with_plugin.utils import load_image, rgb_to_grayscale


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

# this func is from end2end.py
def normalize_keypoints(
    kpts: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    size = torch.tensor([w, h], dtype=torch.float32, device=kpts.device)
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        # nargs="+",
        type=list_of_ints,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers. Example : --img_size 256,512",
    )
    parser.add_argument(
        "--superpoint_onnx_path", # originally is --extractor_path
        type=str,
        default=None,
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )
    parser.add_argument(
        "--lightglue_onnx_path", # originally is --lightglue_path
        type=str,
        default=None,
        required=False,
        help="Path to save the LightGlue ONNX model.",
    )
    # Extractor-specific args:
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=None,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()


def export_onnx(
    img_size=512,
    # extractor_type="superpoint",  # delete
    superpoint_onnx_path=None,
    lightglue_onnx_path=None,
    img_0_path="./assets/sacre_coeur1.jpg",
    img_1_path="./assets/sacre_coeur2.jpg",
    # end2end=False,    # delete
    # dynamic=False,    # delete
    max_num_keypoints=None,
):
    # Handle args
    if isinstance(img_size, List) and len(img_size) == 1:
        img_size = img_size[0]

    max_num_keypoints = max_num_keypoints if max_num_keypoints is not None else 1024

    # Sample images for tracing
    image_0, scales_0 = load_image(img_0_path, resize=img_size)
    image_1, scales_1 = load_image(img_1_path, resize=img_size)

    # SuperPoint works on grayscale images.
    image_0 = rgb_to_grayscale(image_0)
    image_1 = rgb_to_grayscale(image_1)
    superpoint = SuperPoint().eval()
    grid_sampler = SuperPointGridSampler().eval()
    lightglue = LightGlue("superpoint").eval()

    with torch.no_grad():
        # batch_dim has to be 1 for images with different sizes, 
        # or we can write a non-one batch_dim version for images with same size, then we have to put batch_dim into dynamic_axes.
        if superpoint_onnx_path is not None:
            torch.onnx.export(
                superpoint,
                image_0[None],
                superpoint_onnx_path,
                input_names=["image"],                  # (1, 1, image_h_dim, image_w_dim)
                output_names=["scores", "descriptors"], # (1, scores_h_dim, scores_w_dim), (1, 256, descriptors_h_dim, descriptors_w_dim)
                opset_version=17,
                dynamic_axes={
                    "image"       : {0: "batch", 2: "image_h_dim", 3: "image_w_dim"},
                    "scores"      : {0: "batch", 1: "scores_h_dim", 2: "scores_w_dim"},
                    "descriptors" : {0: "batch", 2: "descriptors_h_dim", 3: "descriptors_w_dim"},
                },
            )

        # image_0
        feats_0 = superpoint(image_0[None])
        scores_0, descriptors_0 = feats_0
        print("image_0.shape       : ", image_0.shape)
        print("scores_0.shape      : ", scores_0.shape)
        print("descriptors_0.shape : ", descriptors_0.shape)
        scores_0 = scores_0.squeeze()
        scores_0[ :4, :] = -1
        scores_0[-4:, :] = -1
        scores_0[:,  :4] = -1
        scores_0[:, -4:] = -1
        mask_gt_threshold_0 = torch.where(scores_0 > 0.0005, 1, 0)
        scores_gt_threshold_0 = torch.index_select(scores_0.view(-1), 0, mask_gt_threshold_0.view(-1).nonzero(as_tuple=False).squeeze())
        keypoints_gt_threshold_0 = mask_gt_threshold_0.squeeze().nonzero(as_tuple=False).flip(1)    # before flip, (row_idx, col_idx), after flip, (col_idx, row_idx)
        k = max_num_keypoints if scores_gt_threshold_0.shape[0] > max_num_keypoints else scores_gt_threshold_0.shape[0]
        scores_topk_0, indices_topk_0 = torch.topk(scores_gt_threshold_0, k=k, dim=0, sorted=False)
        keypoints_topk_0 = torch.index_select(keypoints_gt_threshold_0, 0, indices_topk_0)
        mask_gt_threshold_0      = mask_gt_threshold_0.unsqueeze(0)
        scores_gt_threshold_0    = scores_gt_threshold_0.unsqueeze(0)
        keypoints_gt_threshold_0 = keypoints_gt_threshold_0.unsqueeze(0)
        scores_topk_0    = scores_topk_0.unsqueeze(0)
        indices_topk_0   = indices_topk_0.unsqueeze(0)
        keypoints_topk_0 = keypoints_topk_0.unsqueeze(0)
        print("mask_gt_threshold_0.shape      : ", mask_gt_threshold_0.shape)
        print("scores_gt_threshold_0.shape    : ", scores_gt_threshold_0.shape)
        print("keypoints_gt_threshold_0.shape : ", keypoints_gt_threshold_0.shape)
        print("scores_topk_0.shape            : ", scores_topk_0.shape)
        print("indices_topk_0.shape           : ", indices_topk_0.shape)
        print("keypoints_topk_0.shape         : ", keypoints_topk_0.shape)
        print("-" * 100)

        # image_1
        feats_1 = superpoint(image_1[None])
        scores_1, descriptors_1 = feats_1
        print("image_1.shape       : ", image_1.shape)
        print("scores_1.shape      : ", scores_1.shape)
        print("descriptors_1.shape : ", descriptors_1.shape)
        scores_1 = scores_1.squeeze()
        scores_1[ :4, :] = -1
        scores_1[-4:, :] = -1
        scores_1[:,  :4] = -1
        scores_1[:, -4:] = -1
        mask_gt_threshold_1 = torch.where(scores_1 > 0.0005, 1, 0)
        scores_gt_threshold_1 = torch.index_select(scores_1.view(-1), 0, mask_gt_threshold_1.view(-1).nonzero(as_tuple=False).squeeze())
        keypoints_gt_threshold_1 = mask_gt_threshold_1.squeeze().nonzero(as_tuple=False).flip(1)    # before flip, (row_idx, col_idx), after flip, (col_idx, row_idx)
        k = max_num_keypoints if scores_gt_threshold_1.shape[0] > max_num_keypoints else scores_gt_threshold_1.shape[0]
        scores_topk_1, indices_topk_1 = torch.topk(scores_gt_threshold_1, k=k, dim=0, sorted=False)
        keypoints_topk_1 = torch.index_select(keypoints_gt_threshold_1, 0, indices_topk_1)
        mask_gt_threshold_1      = mask_gt_threshold_1.unsqueeze(0)
        scores_gt_threshold_1    = scores_gt_threshold_1.unsqueeze(0)
        keypoints_gt_threshold_1 = keypoints_gt_threshold_1.unsqueeze(0)
        scores_topk_1    = scores_topk_1.unsqueeze(0)
        indices_topk_1   = indices_topk_1.unsqueeze(0)
        keypoints_topk_1 = keypoints_topk_1.unsqueeze(0)
        print("mask_gt_threshold_1.shape      : ", mask_gt_threshold_1.shape)
        print("scores_gt_threshold_1.shape    : ", scores_gt_threshold_1.shape)
        print("keypoints_gt_threshold_1.shape : ", keypoints_gt_threshold_1.shape)
        print("scores_topk_1.shape            : ", scores_topk_1.shape)
        print("indices_topk_1.shape           : ", indices_topk_1.shape)
        print("keypoints_topk_1.shape         : ", keypoints_topk_1.shape)
        print("-" * 100)

        descriptors_topk_0 = grid_sampler(keypoints_topk_0, descriptors_0)
        descriptors_topk_1 = grid_sampler(keypoints_topk_1, descriptors_1)

        print("descriptors_topk_0.shape : ", descriptors_topk_0.shape)
        print("descriptors_topk_1.shape : ", descriptors_topk_1.shape)
        print("-" * 100)

        keypoints_topk_normalized_0 = normalize_keypoints(
            keypoints_topk_0.squeeze(0),    # (col_idx, row_idx)
            image_0.shape[1],               # h
            image_0.shape[2]                # w
        ).unsqueeze(0)
        keypoints_topk_normalized_1 = normalize_keypoints(
            keypoints_topk_1.squeeze(0), 
            image_1.shape[1], 
            image_1.shape[2]
        ).unsqueeze(0)

        print("keypoints_topk_normalized_0.shape : ", keypoints_topk_normalized_0.shape)
        print("keypoints_topk_normalized_1.shape : ", keypoints_topk_normalized_1.shape)
        print("-" * 100)

        if lightglue_onnx_path is not None:
            torch.onnx.export(
                lightglue,
                (keypoints_topk_normalized_0, keypoints_topk_normalized_1, descriptors_topk_0, descriptors_topk_1),
                lightglue_onnx_path,
                input_names=["keypoints_0", "keypoints_1", "descriptors_0", "descriptors_1"],
                output_names=["lightglue_descriptors_0", "lightglue_descriptors_1", "lightglue_scores"],
                opset_version=16,   # my trt doesn't support onnx-opset-17's layernorm
                dynamic_axes={
                    "keypoints_0"             : {1: "num_keypoints_0"},
                    "keypoints_1"             : {1: "num_keypoints_1"},
                    "descriptors_0"           : {1: "num_keypoints_0"},
                    "descriptors_1"           : {1: "num_keypoints_1"},
                    "lightglue_descriptors_0" : {1: "num_keypoints_0"},
                    "lightglue_descriptors_1" : {1: "num_keypoints_1"},
                    "lightglue_scores"        : {1: "num_keypoints_0", 2: "num_keypoints_1"},
                }
            )

        # batch_dim has to be 1
        lightglue_descriptors_0, lightglue_descriptors_1, lightglue_scores = lightglue(
            keypoints_topk_normalized_0, 
            keypoints_topk_normalized_1, 
            descriptors_topk_0, 
            descriptors_topk_1
        )
        print("lightglue_descriptors_0.shape : ", lightglue_descriptors_0.shape, lightglue_descriptors_0.dtype)  # (1, num_keypoints_0, 256)
        print("lightglue_descriptors_1.shape : ", lightglue_descriptors_1.shape, lightglue_descriptors_1.dtype)  # (1, num_keypoints_1, 256)
        print("lightglue_scores.shape        : ", lightglue_scores.shape, lightglue_scores.dtype)                # (1, num_keypoints_0, num_keypoints_1)
        print("-" * 100)


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
