import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is None:
        return
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process_image(image):
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi_image = region_of_interest(canny_image, vertices)
    
    lines = cv2.HoughLinesP(roi_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
    image_with_lines = draw_lines(image, lines)
    
    return image_with_lines

def main():
    cap = cv2.VideoCapture('path_to_video.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_lanes = process_image(frame)
        cv2.imshow('Lane Detection', frame_with_lanes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()