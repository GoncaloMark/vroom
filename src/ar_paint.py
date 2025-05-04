import cv2
import numpy as np
import json
import argparse
from typing import Any, Optional, Tuple, Dict
import datetime
import os
import pynng
import mediapipe as mp
import asyncio

def parse_arguments() -> Dict[str, Any]:
    """
    Parses command-line arguments.

    Returns:
        dict: A dictionary with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="AR painting software with optional shake prevention and mouse control.")
    parser.add_argument("-j", "--json", type=str, help="Full path to JSON file.")
    parser.add_argument("--use_shake_prevention", action="store_true", help="Enable shake prevention functionality.")
    parser.add_argument("--use_mouse", action="store_true", help="Use mouse position for drawing instead of color.")
    parser.add_argument("--voice", action="store_true", help="Use voice commands.")
    parser.add_argument("--gestures", action="store_true", help="Use gesture commands.")
    parser.add_argument("--num_paint", type=str, help="Paint with numbers image path.")
    return vars(parser.parse_args())

class GestureDetector:
    def __init__(self):
        """
        Initializes GestureDetector instance.

        Attributes:
            hand_positions (list): List of hand positions.
            paint_mode (bool): Flag for paint mode.
            mp_hands (mediapipe.solutions.hands.Hands): MediaPipe hand detection instance.
            hands (mediapipe.solutions.hands.Hands): MediaPipe hands detection.
        """
        self.hand_positions = []
        self.paint_mode = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    async def detect_gesture(self, frame: np.ndarray) -> None:
        """
        Detects gestures from a given frame.

        Args:
            frame (np.ndarray): The current video frame.

        Updates:
            hand_positions: List of positions of hand landmarks.
            paint_mode: Whether the gesture indicates paint mode.
        """
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.hand_positions = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]

            self.paint_mode = self.is_two_fingers_up()

    def is_two_fingers_up(self) -> bool:
        """
        Checks if only the index and middle fingers are up.

        Returns:
            bool: True if the index and middle fingers are up, False otherwise.
        """
        if len(self.hand_positions) < 21:
            return False
        index_tip, middle_tip = self.hand_positions[8], self.hand_positions[12]
        
        return (index_tip[1] < self.hand_positions[6][1] and  
                middle_tip[1] < self.hand_positions[10][1] and  
                self.hand_positions[16][1] > self.hand_positions[14][1] and  
                self.hand_positions[20][1] > self.hand_positions[18][1])


class ARPaint:
    def __init__(self, json_file: str, use_shake_prevention: bool = False, use_mouse: bool = False, num_paint: str = None, voice_enabled: bool = False, gestures_enabled:bool = False, detector: GestureDetector = None) -> None:
        """
        Initializes ARPaint instance.

        Args:
            json_file (str): Path to the JSON configuration file.
            use_shake_prevention (bool): Whether to use shake prevention.
            use_mouse (bool): Whether to use mouse input for painting.
            num_paint (str): Path to image used for painting with numbers.
            voice_enabled (bool): Whether to enable voice commands.
            gestures_enabled (bool): Whether to enable gesture controls.
            detector (GestureDetector): Gesture detector for recognizing hand gestures.

        Attributes:
            cap (cv2.VideoCapture): Video capture instance.
            paint_size (int): Size of the paintbrush.
            paint_color (Tuple[int, int, int]): Color of the paintbrush.
            is_painting (bool): Flag to check if the user is painting.
            last_centroid (Optional[Tuple[int, int]]): Last centroid position used for painting.
            shake_threshold (int): Threshold to detect shake movements.
            canvas (np.ndarray): Canvas where the painting occurs.
            drawing_layer (np.ndarray): Layer for drawing on top of the canvas.
            accuracy (float): Accuracy of the painting relative to the reference image.
            delete (bool): Flag for delete mode.
            voice_enabled (bool): Whether voice commands are enabled.
            command (Optional[str]): Voice command received.
            gestures (bool): Whether gesture commands are enabled.
        """
        
        self.cap = cv2.VideoCapture(0)
        self.paint_size = 5
        self.paint_color = (0, 0, 255)  # Start with red
        self.is_painting = False
        self.last_centroid = None
        self.centroid_size = 50
        self.shake_threshold = 50
        self.use_shake_prevention = use_shake_prevention
        self.use_mouse = use_mouse
        self.shape_in_progress = False
        self.shape_start_point = None
        self.circle_mode = False
        self.square_mode = False
        self.video_mode = False
        self.num_paint_path = num_paint
        self.to_paint = None
        self.painted_img_path = f"{os.path.splitext(num_paint)[0]}_paint.png" if num_paint else None
        self.accuracy = 0.0
        self.delete = False
        self.voice_enabled = voice_enabled
        self.command = None
        self.gestures = gestures_enabled
        if voice_enabled:
            self.msg_client = pynng.Sub0(dial="tcp://127.0.0.1:5555")
            self.msg_client.subscribe(b"")

        if gestures_enabled:
            self.gesture_detector = detector

        # Persistent canvas for painting
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imshow("Paint Window", self.canvas)  # Open window before setting callback
        self.drawing_layer = np.zeros((480, 640, 3), dtype=np.uint8) 

        self.to_save = self.canvas.copy

        if self.num_paint_path != '' and self.num_paint_path is not None:
            print(self.num_paint_path)
            self.to_paint = cv2.imread(self.num_paint_path)

        if self.use_mouse:
            cv2.setMouseCallback("Paint Window", self.update_mouse_position)
            self.mouse_position = (0, 0)

        self.show_palette = False
        self.palette_colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 0),    # Dark Blue
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Red
        ]

        if json_file is not None:
            try:
                self.color_limits = self.load_json_limits(json_file)
            except Exception as e:
                raise ValueError(f"Error loading limits JSON file: {e}")

            self.color_space = 'HSV' if 'H' in self.color_limits else 'BGR'
        else:
            self.color_space = 'HSV'
        print(f"Using {self.color_space} color space")

    @staticmethod
    def load_json_limits(json_file: str) -> Dict[str, Dict[str, int]]:
        """
        Loads color limits from a JSON file.

        Args:
            json_file (str): Path to the JSON file containing the color limits.

        Returns:
            dict: A dictionary of color limits.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data['limits']

    def update_mouse_position(self, event, x, y, flags, param):
        """
        Updates mouse position during drawing.

        Args:
            event (int): Type of mouse event.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags (int): Flags associated with the mouse event.
            param (Any): Additional parameters.
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.show_palette:
                self.handle_palette_click(x, y)
            else:
                self.is_painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self.num_paint_path is not None:
                self.evaluate_painting()
            self.is_painting = False
            self.last_centroid = None
            if self.delete:
                self.delete = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Process a frame from the camera feed, applies color space conversion, and identifies the largest object in the frame.

        Args:
            frame (np.ndarray): The input frame to be processed.

        Returns:
            Tuple[np.ndarray, Optional[Tuple[int, int]]]: 
                - The processed frame with the detected object highlighted.
                - The centroid of the largest object, if found, else None.
        """
        frame = cv2.flip(frame, 1)
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if self.color_space == 'HSV' else frame
        lower_bound = np.array([self.color_limits[channel]['min'] for channel in self.color_limits])
        upper_bound = np.array([self.color_limits[channel]['max'] for channel in self.color_limits])
        mask = cv2.inRange(color_frame, lower_bound, upper_bound)
        largest_mask, centroid = self.find_largest_object(mask)
        
        if largest_mask is not None:
            frame[largest_mask > 0] = frame[largest_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        return frame, centroid

    @staticmethod
    def find_largest_object(mask: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Identifies the largest connected component in a binary mask and returns its mask and centroid.

        Args:
            mask (np.ndarray): A binary mask of the frame.

        Returns:
            Optional[Tuple[np.ndarray, Tuple[int, int]]]:
                - The mask of the largest object.
                - The centroid coordinates of the largest object, or None if no object is found.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_mask = np.zeros_like(mask)
            largest_mask[labels == largest_label] = 255
            largest_centroid = tuple(map(int, centroids[largest_label]))
            return largest_mask, largest_centroid
        
        return None, None

    def draw_big_cross(self, frame: np.ndarray, centroid: Tuple[int, int], size: int = 50, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        """
        Draws a big cross at the specified centroid on the frame to indicate the location of the object.

        Args:
            frame (np.ndarray): The frame on which the cross will be drawn.
            centroid (Tuple[int, int]): The coordinates (x, y) of the centroid to draw the cross at.
            size (int, optional): The size of the cross. Default is 50.
            color (Tuple[int, int, int], optional): The color of the cross. Default is green (0, 255, 0).
            thickness (int, optional): The thickness of the cross lines. Default is 2.
        """
        cX, cY = centroid
        cv2.line(frame, (cX, cY - size), (cX, cY + size), color, thickness)
        cv2.line(frame, (cX - size, cY), (cX + size, cY), color, thickness)

    def draw(self, position: Tuple[int, int]) -> None:
        """
        Draws on the drawing layer based on the given position and whether shake prevention is enabled.

        Args:
            position (Tuple[int, int]): The position to draw at (x, y).
        """
        if self.use_shake_prevention and self.last_centroid:
            dist = np.linalg.norm(np.array(position) - np.array(self.last_centroid))
            if dist > self.shake_threshold:
                cv2.circle(self.drawing_layer, position, self.paint_size // 2, self.paint_color, -1)
                print("Shake detected: drawing only a point.")
            else:
                cv2.line(self.drawing_layer, self.last_centroid, position, self.paint_color, self.paint_size)
        elif self.last_centroid:
            cv2.line(self.drawing_layer, self.last_centroid, position, self.paint_color, self.paint_size)
        else:
            cv2.circle(self.drawing_layer, position, self.paint_size // 2, self.paint_color, -1)

        self.last_centroid = position

    def evaluate_painting(self) -> float:
        """
        Evaluates the accuracy of the painting by comparing the painted canvas to a reference image.

        Returns:
            float: The accuracy of the painting as a percentage.
        """
        if self.num_paint_path is not None and self.num_paint_path != '':
            reference = cv2.imread(self.painted_img_path)
            reference = self.overlay_image(np.zeros_like(self.drawing_layer), reference)
            mask = np.any(self.drawing_layer > 0, axis=-1)
            painting = self.overlay_image(self.drawing_layer.copy(), self.to_paint)
            painting[mask] = self.drawing_layer[mask]

            ref_mask = np.any(reference > 0, axis=-1)

            difference = cv2.absdiff(reference[mask], painting[mask])

            if difference is None:
                return 0.0

            matched_pixels = np.sum(np.all(difference < 10, axis=-1))

            total_painted_pixels = np.sum(ref_mask)

            self.accuracy = (matched_pixels / total_painted_pixels) * 100
            self.accuracy = max(0, min(100, self.accuracy))
        return self.accuracy

    def deletion(self, position: Tuple[int, int], size: int = 20) -> None:
        """
        Deletes the drawing in a circular area around the given position on the drawing layer.

        Args:
            position (Tuple[int, int]): The center position of the area to delete.
            size (int, optional): The radius of the circular area to delete. Default is 20.
        """
        x, y = position
        y1, y2 = max(0, y - size), min(self.canvas.shape[0], y + size)
        x1, x2 = max(0, x - size), min(self.canvas.shape[1], x + size)
        
        # From what I understood this creates a grid with bound from x1 to x2 (horizontal) and from y1 to y2 (vertical) so we can then compute the euclidean distance to the center
        Y, X = np.ogrid[y1:y2, x1:x2]
        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist_from_center <= size
        
        self.drawing_layer[y1:y2, x1:x2][mask] = 0

    def overlay_image(self, background, overlay):
        """
        Overlays one image (overlay) onto another (background) using transparency based on the overlay's mask.

        Args:
            background (np.ndarray): The background image.
            overlay (np.ndarray): The image to overlay on top of the background.

        Returns:
            np.ndarray: The combined image with the overlay applied.
        """
        bg_height, bg_width = background.shape[:2]
        overlay_height, overlay_width = overlay.shape[:2]
        x = (bg_width - overlay_width) // 2
        y = (bg_height - overlay_height) // 2

        roi = background[y:y+overlay_height, x:x+overlay_width]

        overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(overlay_gray, 240, 255, cv2.THRESH_BINARY_INV)  
        mask_inv = cv2.bitwise_not(mask)

        background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask)

        combined = cv2.add(background_bg, overlay_fg)
        background[y:y+overlay_height, x:x+overlay_width] = combined

        return background
    
    def draw_color_indicator(self, display: np.ndarray) -> np.ndarray:
        """
        Draws a color indicator and control instructions on the display.

        Args:
            display (np.ndarray): The image to overlay the indicator and instructions on.

        Returns:
            np.ndarray: The display with the color indicator drawn.
        """
        indicator_size = 60
        padding = 10
        
        cv2.rectangle(display, (padding, padding), (padding + indicator_size, padding + indicator_size), (0, 0, 0), 2)
        cv2.rectangle(display, (padding + 2, padding + 2), (padding + indicator_size - 2, padding + indicator_size - 2), self.paint_color, -1)
        
        text_color = (0, 0, 0)
        outline_color = (255, 255, 255)
        
        # White outline
        def put_outlined_text(text, pos_x, pos_y):
            cv2.putText(display, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 3)
            cv2.putText(display, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        put_outlined_text("Current Color", padding, padding + indicator_size + 20)
        put_outlined_text("P: Show Palette", padding, padding + indicator_size + 40)
        
        return display

    def draw_num_paint_legend(self, display: np.ndarray) -> np.ndarray:
        """
        Draws a legend showing the colors used for painting along with the current accuracy.

        Args:
            display (np.ndarray): The image to overlay the legend on.

        Returns:
            np.ndarray: The display with the painting legend drawn.
        """
        if self.num_paint_path:
            legend_start_y = 150
            padding = 10
            
            colors = {
                "1: Red": (0, 0, 255),
                "2: Blue": (255, 0, 0),
                "3: Green": (0, 255, 0)
            }
            
            for i, (text, color) in enumerate(colors.items()):
                y_pos = legend_start_y + (i * 25)
                cv2.rectangle(display, (padding, y_pos), (padding + 20, y_pos + 20), color, -1)
                cv2.rectangle(display, (padding, y_pos), (padding + 20, y_pos + 20), (0, 0, 0), 1)
                
                cv2.putText(display, text, (padding + 30, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                cv2.putText(display, text, (padding + 30, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.putText(display, f"Accuracy: {self.accuracy:.2f}", (padding, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(display, f"Accuracy: {self.accuracy:.2f}", (padding, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return display

    def draw_color_palette(self, display: np.ndarray) -> np.ndarray:
        """
        Draws a color palette on the display for the user to select a color from.

        Args:
            display (np.ndarray): The image to overlay the palette on.

        Returns:
            np.ndarray: The display with the color palette drawn.
        """
        if self.show_palette:
            palette_width = 180
            palette_height = 180
            cell_size = 60
            
            start_x = (display.shape[1] - palette_width) // 2
            start_y = (display.shape[0] - palette_height) // 2
            
            cv2.rectangle(display, (start_x - 10, start_y - 10), (start_x + palette_width + 10, start_y + palette_height + 10), (0, 0, 0), 2)
            
            for i, color in enumerate(self.palette_colors):
                row = i // 3
                col = i % 3
                x1 = start_x + (col * cell_size)
                y1 = start_y + (row * cell_size)
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                cv2.rectangle(display, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 0), 1)

            text = "Click to select color, ESC to close"
            cv2.putText(display, text, (start_x - 5, start_y + palette_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(display, text, (start_x - 5, start_y + palette_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return display
    
    def handle_palette_click(self, x: int, y: int) -> None:
        """
        Handles a click event on the color palette to select a color.

        Args:
            x (int): The x-coordinate of the click.
            y (int): The y-coordinate of the click.
        """
        if self.show_palette:
            palette_width = 180
            palette_height = 180
            cell_size = 60
            
            start_x = (self.canvas.shape[1] - palette_width) // 2
            start_y = (self.canvas.shape[0] - palette_height) // 2
            
            if (start_x <= x <= start_x + palette_width and 
                start_y <= y <= start_y + palette_height):
                
                col = (x - start_x) // cell_size
                row = (y - start_y) // cell_size
                index = row * 3 + col
                
                if 0 <= index < len(self.palette_colors):
                    self.paint_color = self.palette_colors[index]
                    self.show_palette = False

    def process_voice_commands(self) -> Optional[str]:
        """
        Processes voice commands retrieved from a nanomsg client.

        Returns:
            Optional[str]: The command retrieved from nanomsg, or None if no command is available.
        """
        try:
            msg = self.msg_client.recv(block = False)
            return json.loads(msg.decode())["command"]
        except pynng.exceptions.TryAgain:
            # Return None if timeout occurs
            return None
        except Exception as e:
            print(f"Error receiving command: {e}")
            return None

    async def run(self) -> None:
        """
        Runs the main loop of the application, processing frames, gestures, voice commands, and updating the display.

        This is an asynchronous method that continuously processes camera frames and executes tasks based on the mode (e.g., mouse, gestures, etc.).

        It also handles displaying the canvas, color palette, and other UI elements.
        """
        while True:
            if self.voice_enabled:
                self.command = self.process_voice_commands()
            ret, frame = self.cap.read()
            if not ret:
                print("Error getting image frame.")
                break

            if self.use_mouse:
                position = self.mouse_position
                frame = cv2.flip(frame, 1)
            elif self.gestures:
                frame = cv2.flip(frame, 1)
                await self.gesture_detector.detect_gesture(frame)
                # print("Didnt block gesture detection")

                # Check gesture modes
                self.is_painting = self.gesture_detector.paint_mode
                if self.gesture_detector.hand_positions:
                    position = self.gesture_detector.hand_positions[8]
                else:
                    position = None
            else:
                frame, position = self.process_frame(frame)

            if self.video_mode:
                display = frame
                display = cv2.add(display, self.drawing_layer)

                self.to_save = frame
                self.to_save = cv2.add(display, self.drawing_layer)
            else:
                display = self.canvas.copy()
                mask = np.any(self.drawing_layer > 0, axis=-1)
                display[mask] = self.drawing_layer[mask] 

                self.to_save = self.canvas.copy()
                mask = np.any(self.drawing_layer > 0, axis=-1)
                self.to_save[mask] = self.drawing_layer[mask] 

            if self.to_paint is not None:
                display = self.overlay_image(display, self.to_paint)
                self.to_save = self.overlay_image(self.to_save, self.to_paint)

            display = self.draw_color_indicator(display)
            display = self.draw_num_paint_legend(display)
            display = self.draw_color_palette(display)

            temp_layer = self.drawing_layer.copy()

            if position:
                if self.is_painting and not self.shape_in_progress:
                    if self.delete:
                        self.deletion(position)
                    else:
                        self.draw(position)
                elif self.shape_in_progress:
                    if self.circle_mode:
                        radius = int(np.linalg.norm(np.array(position) - np.array(self.shape_start_point)))
                        cv2.circle(temp_layer, self.shape_start_point, radius, self.paint_color, 1)
                    elif self.square_mode:
                        side_length = int(np.linalg.norm(np.array(position) - np.array(self.shape_start_point)))
                        top_left = (min(self.shape_start_point[0], position[0]), min(self.shape_start_point[1], position[1]))
                        cv2.rectangle(temp_layer, top_left, (top_left[0] + side_length, top_left[1] + side_length), self.paint_color, 1)

                if position:
                    self.draw_big_cross(display, position, size=50, color=(0, 0, 255), thickness=2)

            cv2.imshow("Paint Window", display)

            key = cv2.waitKey(1) & 0xFF
            self.handle_keypress(key, position)

    def handle_keypress(self, key: int, position: Optional[Tuple[int, int]]) -> None:
        if key == ord('q') or self.command == "stop":
            if self.voice_enabled:
                self.msg_client.close()
            self.cap.release()
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('r') or self.command == "red":
            self.paint_color = (0, 0, 255)  # Red
        elif key == ord('g') or self.command == "green":
            self.paint_color = (0, 255, 0)  # Green
        elif key == ord('b') or self.command == "blue":
            self.paint_color = (255, 0, 0)  # Blue
        elif key == ord('p'):
            self.show_palette = not self.show_palette
        elif key == 27: 
            self.show_palette = False
        elif key == ord('+'):
            self.paint_size += 1
            self.centroid_size += 5
        elif key == ord('-'):
            self.paint_size = max(1, self.paint_size - 1)
            self.centroid_size = max(5, self.centroid_size - 5)
        elif key == ord('c') or self.command == "clear":
            self.clear_canvas()
        elif key == ord('w'):
            self.save_image()
        elif key == ord('e'):
            score = self.evaluate_painting()
            print(f"Your painting scored: {score}")
        elif key == ord('d') or self.command == "delete":
            self.delete = not self.delete
        elif key == ord(' '):
            if not self.use_mouse:
                self.is_painting = not self.is_painting 
                self.last_centroid = None
                self.evaluate_painting()
        elif key == ord('v'):
            self.video_mode = not self.video_mode
        elif key == ord('o'):  # Circle mode
            if not self.shape_in_progress:
                self.shape_start_point = position
                self.circle_mode = True
                self.shape_in_progress = True
            else:
                radius = int(np.linalg.norm(np.array(position) - np.array(self.shape_start_point)))
                cv2.circle(self.drawing_layer, self.shape_start_point, radius, self.paint_color, -1)
                self.shape_in_progress = False
                self.circle_mode = False
        elif key == ord('s'):  # Square mode
            if not self.shape_in_progress:
                self.shape_start_point = position
                self.square_mode = True
                self.shape_in_progress = True
            else:
                side_length = int(np.linalg.norm(np.array(position) - np.array(self.shape_start_point)))
                top_left = (min(self.shape_start_point[0], position[0]),
                            min(self.shape_start_point[1], position[1]))
                cv2.rectangle(self.drawing_layer, top_left, (top_left[0] + side_length, top_left[1] + side_length), self.paint_color, -1)
                self.shape_in_progress = False
                self.square_mode = False

    def clear_canvas(self) -> None:
        self.canvas = np.ones_like(self.canvas) * 255
        self.drawing_layer = np.zeros_like(self.drawing_layer)
        print("Canvas cleared.")

    def save_image(self) -> None:
        if not os.path.exists("images"):
            os.makedirs("images")
        filename = datetime.datetime.now().strftime("images/%Y%m%d_%H%M%S.png")
        cv2.imwrite(filename, self.to_save)
        print(f"Image saved as {filename}")

async def main():
    args = parse_arguments()
    detector = GestureDetector()
    ar_paint = ARPaint(args['json'], args['use_shake_prevention'], args['use_mouse'], args['num_paint'], args['voice'], args['gestures'], detector)
    await ar_paint.run()

if __name__ == "__main__":
    asyncio.run(main())
