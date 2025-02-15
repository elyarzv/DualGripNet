# gqcnn-master/examples/policy.py

import json
import os
import time
import numpy as np
from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode

# Set up logger.
logger = Logger.get_logger("examples/policy.py")


def run_grasping_policy(model_name, depth_image, segmask, camera_intr, model_dir, config_filename, fully_conv):
    # assert not (fully_conv and depth_im_filename is not None
    #             and segmask_filename is None
    #             ), "Fully-Convolutional policy expects a segmask."

    # if depth_im_filename is None:
    #     if fully_conv:
    #         depth_im_filename = os.path.join(
    #             os.path.dirname(os.path.realpath(__file__)), "..",
    #             "data/examples/clutter/primesense/depth_0.npy")
    #     else:
    #         depth_im_filename = os.path.join(
    #             os.path.dirname(os.path.realpath(__file__)), "..",
    #             "data/examples/single_object/primesense/depth_0.npy")
    # if fully_conv and segmask_filename is None:
    #     segmask_filename = os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)), "..",
    #         "data/examples/clutter/primesense/segmask_0.png")
    # if camera_intr_filename is None:
    #     camera_intr_filename = os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)), "..",
    #         "data/calib/primesense/primesense.intr")

    # # Set model if provided.
    # if model_dir is None:
    #     model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                              "../models")
    
    model_path = os.path.join(model_dir, model_name)

    # Get configs.
    model_config = json.load(open(os.path.join(model_path, "config.json"), "r"))
    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError("Input data mode {} not supported!".format(input_data_mode))

    # Set config.
    if config_filename is None:
        if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW or gripper_mode == GripperMode.PARALLEL_JAW):
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "cfg/examples/fc_gqcnn_pj.yaml")
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "cfg/examples/gqcnn_pj.yaml")
        elif (gripper_mode == GripperMode.LEGACY_SUCTION or gripper_mode == GripperMode.SUCTION):
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "cfg/examples/fc_gqcnn_suction.yaml")
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "cfg/examples/gqcnn_suction.yaml")

    # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr)

    # Read images.
    depth_data = depth_image
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8), frame=camera_intr.frame)

    # Optionally read a segmask.
    segmask = None
    if segmask is not None:
        segmask = BinaryImage.open(segmask)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)

    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

    if "input_images" in policy_config["vis"] and policy_config["vis"]["input_images"]:
        vis.figure(size=(10, 10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
        vis.subplot(1, num_plot, 1)
        vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1, num_plot, 2)
            vis.imshow(segmask)
        vis.show()

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # Set input sizes for fully-convolutional policy.
    if fully_conv:
        policy_config["metric"]["fully_conv_gqcnn_config"]["im_height"] = depth_im.shape[0]
        policy_config["metric"]["fully_conv_gqcnn_config"]["im_width"] = depth_im.shape[1]

    # Init policy.
    if fully_conv:
        if policy_config["type"] == "fully_conv_suction":
            policy = FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config["type"] == "fully_conv_pj":
            policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
        else:
            raise ValueError("Invalid fully-convolutional policy type: {}".format(policy_config["type"]))
    else:
        policy_type = "cem"
        if "type" in policy_config:
            policy_type = policy_config["type"]
        if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))

    # Query policy.
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))

    # Vis final grasp.
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth, vmin=policy_config["vis"]["vmin"], vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(action.grasp.depth, action.q_value))
        vis.show()

    return action


# Example call to the function (this part would be in your calling script)
if __name__ == "__main__":
    action = run_grasping_policy(
        model_name="GQCNN-4.0-PJ",
        depth_image="path/to/depth_0.npy",
        segmask="path/to/segmask_0.png",
        camera_intr="path/to/camera_intrinsics.intr",
        model_dir="path/to/models",
        config_filename=None,
        fully_conv=True
    )
    print("Planned Action:", action)
