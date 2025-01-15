import pyzed.sl as sl
import cv2
import numpy as np

# Initialize the ZED camera
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Select the resolution
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use the neural depth mode
init_params.coordinate_units = sl.UNIT.METER  # Set the units to meters

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit()

# Enable positional tracking for semantic segmentation
positional_tracking_parameters = sl.PositionalTrackingParameters()
if zed.enable_positional_tracking(positional_tracking_parameters) != sl.ERROR_CODE.SUCCESS:
    print("Failed to enable positional tracking")
    exit()

# Enable the semantic segmentation module
semantic_params = sl.ObjectDetectionParameters()
semantic_params.enable_tracking = False  # Set to True if you want to track objects over time
semantic_params.enable_mask_output = True  # Enable output of segmentation masks

if zed.enable_object_detection(semantic_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to enable semantic segmentation")
    exit()

# Capture one frame
runtime_params = sl.RuntimeParameters()
if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # Retrieve the image
    image = sl.Mat()
    zed.retrieve_image(image, sl.VIEW.LEFT)

    # Retrieve the semantic segmentation mask
    mask = sl.Mat()
    zed.retrieve_objects(mask, sl.MEASURE.MASK)

    # Convert the ZED image and mask to numpy arrays
    image_data = image.get_data()
    mask_data = mask.get_data()

    # Visualize the original image
    cv2.imshow("Captured Image", image_data)

    # Visualize the semantic segmentation mask
    mask_display = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
    cv2.imshow("Semantic Segmentation Mask", mask_display)

    # Save the image and mask
    cv2.imwrite("captured_image.png", image_data)
    cv2.imwrite("semantic_segmentation_mask.png", mask_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Close the camera
zed.close()
