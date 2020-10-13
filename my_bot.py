import cv2
import numpy as np
from PIL import ImageGrab
import win32gui

# Detect the window with Tetris game
windows_list = []
toplist = []
def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))


win32gui.EnumWindows(enum_win, toplist)

# Game handle
game_hwnd = 0
for (hwnd, win_text) in windows_list:
    if "Task Manager" in win_text:
        game_hwnd = hwnd
#auto-py-to-exe convert to exe
while True:
    position = win32gui.GetWindowRect(game_hwnd)
    # Take screenshot
    screenshot = ImageGrab.grab(position)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cv2.imshow("Screen", screenshot)
    if cv2.waitKey(21) & 0xFF == ord('q'):
        break;