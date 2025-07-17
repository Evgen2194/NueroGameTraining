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
            cv2.circle(overlay, (center_x, center_y), 50, (0, 255, 0), 5)
            cv2.putText(overlay, "CLICK", (center_x - 50, center_y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        elif action_type == config.ACTION_DRAG:
            params = action_params[0].cpu().numpy()
            start_x = int(params[0] * vis_frame.shape[1])
            start_y = int(params[1] * vis_frame.shape[0])
            end_x = int(params[2] * vis_frame.shape[1])
            end_y = int(params[3] * vis_frame.shape[0])
            cv2.arrowedLine(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), 10)
            cv2.putText(overlay, "DRAG", (start_x - 50, start_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        elif action_type == config.ACTION_WAIT:
            cv2.putText(overlay, "WAIT", (vis_frame.shape[1] // 3, vis_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

        if self.action_visual is None:
            self.action_visual = "Agent Suggestion"
            cv2.namedWindow(self.action_visual, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.action_visual, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(self.action_visual, cv2.WND_PROP_TOPMOST, 1)


        cv2.imshow(self.action_visual, vis_frame)
        cv2.waitKey(1)

    def get_user_feedback(self):
        self.feedback = None
        with keyboard.Listener(on_press=self._on_press) as listener:
            listener.join()
        return self.feedback

    def _on_press(self, key):
        if key == keyboard.Key.up:
            self.feedback = "approve"
        elif key == keyboard.Key.down:
            self.feedback = "reject"
        elif key == keyboard.Key.left:
            self.feedback = "force_wait"
        elif key == keyboard.Key.space:
            self.feedback = "skip"
        elif hasattr(key, 'char'):
            if key.char == '1':
                self.feedback = "start_mission"
            elif key.char == '2':
                self.feedback = "mission_win"
            elif key.char == '3':
                self.feedback = "mission_loss"
        # Stop the listener regardless of the key
        return False

    def perform_action(self, action):
        action_type, action_params = action
        action_type = action_type.item()

        # Get absolute coordinates based on game_area
        abs_x = self.game_area[0]
        abs_y = self.game_area[1]
        width = self.game_area[2] - self.game_area[0]
        height = self.game_area[3] - self.game_area[1]

        if action_type == config.ACTION_CLICK:
            params = action_params[0].cpu().numpy()
            target_x = abs_x + int(params[0] * width)
            target_y = abs_y + int(params[1] * height)
            pyautogui.click(target_x, target_y)
            print(f"Action: CLICK at ({target_x}, {target_y})")

        elif action_type == config.ACTION_DRAG:
            params = action_params[0].cpu().numpy()
            start_x = abs_x + int(params[0] * width)
            start_y = abs_y + int(params[1] * height)
            end_x = abs_x + int(params[2] * width)
            end_y = abs_y + int(params[3] * height)
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=0.5)
            print(f"Action: DRAG from ({start_x}, {start_y}) to ({end_x}, {end_y})")

        elif action_type == config.ACTION_WAIT:
            # No actual action, just wait
            print("Action: WAIT")
            pass

if __name__ == '__main__':
    ui = UI()
    ui.select_game_area()
