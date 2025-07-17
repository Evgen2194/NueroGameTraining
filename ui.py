import cv2
import numpy as np
from pynput import mouse, keyboard
import pyautogui
import config

class UI:
    def __init__(self):
        self.game_area = None
        self.action_visual = None

    def select_game_area(self):
        self.points = []
        print("Select the top-left corner of the game area and click.")

        with mouse.Listener(on_click=self._on_click_area) as listener:
            listener.join()

        print("Select the bottom-right corner of the game area and click.")

        with mouse.Listener(on_click=self._on_click_area) as listener:
            listener.join()

        self.game_area = (self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1])
        print(f"Game area selected: {self.game_area}")

    def _on_click_area(self, x, y, button, pressed):
        if pressed:
            self.points.append((x, y))
            return False

    def display_action_suggestion(self, screenshot, action):
        action_type, action_params = action
        action_type = action_type.item()

        # Create a copy for drawing
        vis_frame = screenshot.copy()
        overlay = vis_frame.copy()

        if action_type == config.ACTION_CLICK:
            params = action_params[0].cpu().numpy()
            center_x = int(params[0] * vis_frame.shape[1])
            center_y = int(params[1] * vis_frame.shape[0])
            cv2.circle(overlay, (center_x, center_y), 20, (0, 255, 0), -1)
        elif action_type == config.ACTION_DRAG:
            params = action_params[0].cpu().numpy()
            start_x = int(params[0] * vis_frame.shape[1])
            start_y = int(params[1] * vis_frame.shape[0])
            end_x = int(params[2] * vis_frame.shape[1])
            end_y = int(params[3] * vis_frame.shape[0])
            cv2.arrowedLine(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
        elif action_type == config.ACTION_WAIT:
            cv2.putText(overlay, "WAIT", (vis_frame.shape[1] // 4, vis_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

        cv2.imshow("Agent Suggestion", vis_frame)
        cv2.waitKey(1) # Necessary to display the window

    def get_user_feedback(self):
        self.feedback = None
        with keyboard.Listener(on_press=self._on_press) as listener:
            listener.join()
        return self.feedback

    def _on_press(self, key):
        try:
            if key.char == '1':
                self.feedback = "start_mission"
                return False
            elif key.char == '2':
                self.feedback = "mission_win"
                return False
            elif key.char == '3':
                self.feedback = "mission_loss"
                return False
        except AttributeError:
            if key == keyboard.Key.up:
                self.feedback = "approve"
                return False
            elif key == keyboard.Key.down:
                self.feedback = "reject"
                return False
            elif key == keyboard.Key.left:
                self.feedback = "force_wait"
                return False
            elif key == keyboard.Key.space:
                self.feedback = "skip"
                return False

    def perform_action(self, action):
        # TODO: Implement action execution
        pass
