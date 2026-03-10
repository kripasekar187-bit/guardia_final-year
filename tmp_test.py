import builtins
builtins.input = lambda prompt: "normal"

import cv2
old_imshow = cv2.imshow
cv2.imshow = lambda *args: None

# We want it to run for maybe 10 frames then quit
frame_count = 0
def mock_waitkey(*args):
    global frame_count
    frame_count += 1
    if frame_count > 10:
        return ord('q')
    return -1

cv2.waitKey = mock_waitkey

try:
    import runpy
    runpy.run_path('collect data.py')
    print("Script ran successfully.")
except Exception as e:
    import traceback
    with open('error_log.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("Error saved to error_log.txt")
