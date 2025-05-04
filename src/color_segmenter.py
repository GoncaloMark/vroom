#!/usr/bin/env python3

import cv2
import json
import numpy as np
from typing import Tuple, Any, Dict
import argparse


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments.

    Returns:
        A dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Definition of test mode")
    parser.add_argument("-c", "--color_space", type=str, required=True, help='Use BGR or HSV.')
    return vars(parser.parse_args())


class ColorSegmenter:
    def __init__(self, color_space: str = 'HSV') -> None:
        """
        Initializes the ColorSegmenter object.

        Args:
            color_space (str): The color space to use for segmentation. Supported values are 'HSV' and 'BGR'. Default is 'HSV'.

        Attributes:
            cap (cv2.VideoCapture): Video capture object to access the camera.
            window_name (str): Name of the OpenCV window where trackbars and images will be displayed.
            frame (np.ndarray): Stores the current video frame for processing.
            color_space (str): The color space used for color segmentation ('HSV' or 'BGR').
        """
        self.cap = cv2.VideoCapture(0)
        self.window_name = 'Color Segmenter'
        self.frame = None
        self.color_space = color_space.upper()

        if self.color_space not in ['HSV', 'BGR']:
            raise ValueError("Invalid color space. Supported values are 'HSV' and 'BGR'.")
        
        self.create_trackbars()

    def create_trackbars(self) -> None:
        """
        Creates the window with the trackbars for adjusting HSV or BGR values.

        Trackbars for HSV or BGR:
            - 'H Min'/'H Max', 'S Min'/'S Max', 'V Min'/'V Max' for HSV
            - 'B Min'/'B Max', 'G Min'/'G Max', 'R Min'/'R Max' for BGR
        """
        cv2.namedWindow(self.window_name)

        if self.color_space == 'HSV':
            cv2.createTrackbar('H Min', self.window_name, 0, 179, self.on_trackbar)
            cv2.createTrackbar('H Max', self.window_name, 179, 179, self.on_trackbar)
            cv2.createTrackbar('S Min', self.window_name, 0, 255, self.on_trackbar)
            cv2.createTrackbar('S Max', self.window_name, 255, 255, self.on_trackbar)
            cv2.createTrackbar('V Min', self.window_name, 0, 255, self.on_trackbar)
            cv2.createTrackbar('V Max', self.window_name, 255, 255, self.on_trackbar)
        elif self.color_space == 'BGR':
            cv2.createTrackbar('B Min', self.window_name, 0, 255, self.on_trackbar)
            cv2.createTrackbar('B Max', self.window_name, 255, 255, self.on_trackbar)
            cv2.createTrackbar('G Min', self.window_name, 0, 255, self.on_trackbar)
            cv2.createTrackbar('G Max', self.window_name, 255, 255, self.on_trackbar)
            cv2.createTrackbar('R Min', self.window_name, 0, 255, self.on_trackbar)
            cv2.createTrackbar('R Max', self.window_name, 255, 255, self.on_trackbar)

    def on_trackbar(self, _ = None) -> None:
        """
        Callback function gets called when a trackbar value changes.
        """
        pass

    def get_trackbar_values(self) -> Tuple[int, int, int, int, int, int]:
        """
        Gets the current trackbar values for HSV or BGR.

        Returns:
            A tuple containing the current trackbar values for either HSV or BGR.
        """
        if self.color_space == 'HSV':
            h_min = cv2.getTrackbarPos('H Min', self.window_name)
            h_max = cv2.getTrackbarPos('H Max', self.window_name)
            s_min = cv2.getTrackbarPos('S Min', self.window_name)
            s_max = cv2.getTrackbarPos('S Max', self.window_name)
            v_min = cv2.getTrackbarPos('V Min', self.window_name)
            v_max = cv2.getTrackbarPos('V Max', self.window_name)
            return h_min, h_max, s_min, s_max, v_min, v_max
        else:
            b_min = cv2.getTrackbarPos('B Min', self.window_name)
            b_max = cv2.getTrackbarPos('B Max', self.window_name)
            g_min = cv2.getTrackbarPos('G Min', self.window_name)
            g_max = cv2.getTrackbarPos('G Max', self.window_name)
            r_min = cv2.getTrackbarPos('R Min', self.window_name)
            r_max = cv2.getTrackbarPos('R Max', self.window_name)
            return b_min, b_max, g_min, g_max, r_min, r_max

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a frame to segment the color based on the HSV/BGR trackbar values.

        Args:
            frame (np.ndarray): The input frame.

        Returns:
            np.ndarray: The segmented image.
        """
        if self.color_space == 'HSV':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_min, h_max, s_min, s_max, v_min, v_max = self.get_trackbar_values()
            lower_bound = (h_min, s_min, v_min)
            upper_bound = (h_max, s_max, v_max)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            b_min, b_max, g_min, g_max, r_min, r_max = self.get_trackbar_values()
            lower_bound = (b_min, g_min, r_min)
            upper_bound = (b_max, g_max, r_max)
            mask = cv2.inRange(frame, lower_bound, upper_bound)

        segmented_image = cv2.bitwise_and(frame, frame, mask=mask)
        return segmented_image

    def output_json(self, filename: str = 'limits.json') -> None:
        """
        Saves the current HSV/BGR trackbar values to a JSON file.

        Args:
            filename (str): The name of the JSON file. Defaults to 'limits.json'.
        """
        if self.color_space == 'HSV':
            h_min, h_max, s_min, s_max, v_min, v_max = self.get_trackbar_values()
            limits = {
                'H': {'min': h_min, 'max': h_max},
                'S': {'min': s_min, 'max': s_max},
                'V': {'min': v_min, 'max': v_max}
            }
        else:
            b_min, b_max, g_min, g_max, r_min, r_max = self.get_trackbar_values()
            limits = {
                'B': {'min': b_min, 'max': b_max},
                'G': {'min': g_min, 'max': g_max},
                'R': {'min': r_min, 'max': r_max}
            }

        with open(filename, 'w') as f:
            json.dump({'limits': limits}, f)

        print(f"Saved limits to {filename}.")

    def run(self) -> None:
        """
        Main loop to capture video and process frames.
        Handles keybindings for saving and quitting.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing video frame.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow(self.window_name, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                self.output_json()
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_arguments()
    color_segmenter = ColorSegmenter(args['color_space'])
    color_segmenter.run()


if __name__ == "__main__":
    main()

