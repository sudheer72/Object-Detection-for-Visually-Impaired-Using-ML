import cv2
import numpy as np
import time
import speech_recognition as sr
import os
from person_registry import PersonRegistry

print("[SYSTEM] Initializing face recognition system...")
registry = PersonRegistry()
recognizer = sr.Recognizer() # speech recognizer for voice input.

print("[SYSTEM] Loading object detection model...")
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    exit()

try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]
except FileNotFoundError:
    print("[ERROR] coco.names file not found")
    exit()

print("[SYSTEM] Initializing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] Failed to initialize camera")
    exit()

# Constants
KNOWN_WIDTH = 40
FOCAL_LENGTH = 700
MIN_DISTANCE = 30
MAX_DISTANCE = 500

# Tracking parameters optimized for better stability
SMOOTH_FACTOR = 0.7  # Increased smoothing for more stable boxes
MAX_AGE = 30         # Keep tracks longer
MATCH_THRESHOLD = 200 # More generous matching
MIN_OBJECT_SIZE = 25  # Minimum size for objects to track

def preprocess_frame(frame):
    """Enhance frame for better detection with YOLO-tiny"""
    # Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Mild sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def calculate_distance(pixel_width):
    if pixel_width <= 0:
        return None
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return max(MIN_DISTANCE, min(MAX_DISTANCE, distance))

def get_position(x, w, frame_width):
    center_x = x + w/2
    if center_x < frame_width/3:
        return "left"
    elif center_x > 2*frame_width/3:
        return "right"
    return "center"

# Runs YOLO object detection and applies Non-Maximum Suppression (NMS) to remove duplicate detections.
def detect_objects(frame, net, classes):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Lower confidence threshold for all objects
            if confidence > 0.25:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Smaller minimum size for all objects
                if w > 20 and h > 20:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # More lenient NMS for objects
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.2)
    
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            if w > 10 and h > 10:
                detected_objects.append({
                    'label': classes[class_ids[i]],
                    'box': (x, y, w, h),
                    'distance': calculate_distance(w),
                    'position': get_position(x, w, width),
                    'confidence': confidences[i]
                })
    
    return detected_objects

def listen_for_name():
    with sr.Microphone() as source:
        try:
            print("[SYSTEM] Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("[SYSTEM] Microphone ready. Speak your name clearly.")
            registry.speak("Please say your name clearly now")
            
            recognizer.energy_threshold = 300
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            try:
                name = recognizer.recognize_google(audio, language="en-US")
                print(f"[USER] Recognized name: '{name}'")
            except sr.UnknownValueError:
                print("[DEBUG] Google failed, trying Sphinx...")
                name = recognizer.recognize_sphinx(audio)
                print(f"[USER] Sphinx recognized name: '{name}'")
            
            registry.speak(f"Did you say {name}? Nod for yes, shake for no")
            time.sleep(3)
            confirmed = True
            
            if confirmed:
                registry.speak(f"Thank you, {name}")
                return name.lower().strip()
            else:
                registry.speak("Let's try again")
                return None
                
        except sr.UnknownValueError:
            print("[ERROR] Speech not understood")
            registry.speak("Couldn't understand. Try again.")
            return None
        except sr.RequestError as e:
            print(f"[ERROR] Speech service error: {e}")
            registry.speak("Speech service issue.")
            return None
        except sr.WaitTimeoutError:
            print("[ERROR] No speech detected")
            registry.speak("No speech detected. Speak louder.")
            return None
        except Exception as e:
            print(f"[ERROR] Speech recognition error: {e}")
            registry.speak("An error occurred.")
            return None

# Main loop variables
registration_mode = False
last_detection_time = 0
detection_interval = 0.3
object_detection_enabled = True
last_announce_time = 0
announce_interval = 5
frame_count = 0
start_time = time.time()
tracked_persons = []
tracked_objects = []
last_announced_items = []

print("[SYSTEM] Starting main detection loop...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame")
            break

        # Preprocess frame for better detection
        frame = preprocess_frame(frame)
        height, width = frame.shape[:2]
        current_time = time.time()
        
        if current_time - last_detection_time > detection_interval:
            last_detection_time = current_time
            
            if object_detection_enabled:
                detected_objects = detect_objects(frame, net, classes)
                objects = [obj for obj in detected_objects if obj['label'] != 'person']
                persons = [obj for obj in detected_objects if obj['label'] == 'person']
                
                # Clear previous frame's drawings
                frame_clean = frame.copy()
                
                # Process objects with yellow boxes
                new_tracked_objects = []
                for obj in objects:
                    x, y, w, h = obj['box']
                    
                    # Find closest existing tracked object
                    best_match = None
                    min_dist = float('inf')
                    center = (x + w/2, y + h/2)
                    
                    for i, tracked in enumerate(tracked_objects):
                        tx, ty, tw, th = tracked['box']
                        t_center = (tx + tw/2, ty + th/2)
                        dist = ((center[0] - t_center[0])**2 + (center[1] - t_center[1])**2)**0.5
                        
                        if dist < min_dist and dist < MATCH_THRESHOLD:
                            min_dist = dist
                            best_match = i
                    
                    if best_match is not None:
                        # Update existing with smoothing
                        tracked = tracked_objects[best_match]
                        tx, ty, tw, th = tracked['box']
                        
                        new_x = int(tx * (1-SMOOTH_FACTOR) + x * SMOOTH_FACTOR)
                        new_y = int(ty * (1-SMOOTH_FACTOR) + y * SMOOTH_FACTOR)
                        new_w = int(tw * (1-SMOOTH_FACTOR) + w * SMOOTH_FACTOR)
                        new_h = int(th * (1-SMOOTH_FACTOR) + h * SMOOTH_FACTOR)
                        
                        tracked['box'] = (new_x, new_y, new_w, new_h)
                        tracked['distance'] = calculate_distance(new_w)
                        tracked['position'] = get_position(new_x, new_w, width)
                        tracked['age'] = 0
                        new_tracked_objects.append(tracked)
                    else:
                        # Add new object
                        new_tracked_objects.append({
                            'box': (x, y, w, h),
                            'label': obj['label'],
                            'distance': calculate_distance(w),
                            'position': get_position(x, w, width),
                            'age': 0,
                            'confidence': obj['confidence']
                        })
                
                # Age out unmatched objects
                for tracked in tracked_objects:
                    if tracked not in new_tracked_objects:
                        tracked['age'] += 1
                        if tracked['age'] < MAX_AGE:
                            new_tracked_objects.append(tracked)
                
                tracked_objects = new_tracked_objects
                
                # Process persons with green boxes
                new_tracked_persons = []
                for person in persons:
                    x, y, w, h = person['box']
                    face_roi = frame[y:y+h, x:x+w]
                    name = registry.recognize_person(face_roi)
                    
                    # Find the best matching existing person
                    best_match = None
                    min_dist = float('inf')
                    center = (x + w/2, y + h/2)
                    
                    for i, tracked in enumerate(tracked_persons):
                        tx, ty, tw, th = tracked['box']
                        t_center = (tx + tw/2, ty + th/2)
                        dist = ((center[0] - t_center[0])**2 + (center[1] - t_center[1])**2)**0.5
                        
                        # Use both distance and size similarity for matching
                        size_similarity = abs(w - tw) + abs(h - th)
                        if dist < min_dist and dist < MATCH_THRESHOLD and size_similarity < 50:
                            min_dist = dist
                            best_match = i
                    
                    if best_match is not None:
                        tracked = tracked_persons[best_match]
                        # Stronger smoothing for stability
                        new_x = int(tracked['box'][0] * 0.7 + x * 0.3)
                        new_y = int(tracked['box'][1] * 0.7 + y * 0.3)
                        new_w = int(tracked['box'][2] * 0.7 + w * 0.3)
                        new_h = int(tracked['box'][3] * 0.7 + h * 0.3)
                        
                        tracked['box'] = (new_x, new_y, new_w, new_h)
                        tracked['distance'] = calculate_distance(new_w)
                        tracked['position'] = get_position(new_x, new_w, width)
                        tracked['age'] = 0
                        if name:  # Only update name if we got a good recognition
                            tracked['name'] = name
                        new_tracked_persons.append(tracked)
                    else:
                        new_tracked_persons.append({
                            'box': (x, y, w, h),
                            'name': name if name else "person",
                            'distance': calculate_distance(w),
                            'position': get_position(x, w, width),
                            'age': 0
                        })
                
                # Age out unmatched persons
                for tracked in tracked_persons:
                    if tracked not in new_tracked_persons:
                        tracked['age'] += 1
                        if tracked['age'] < MAX_AGE:
                            new_tracked_persons.append(tracked)
                
                tracked_persons = new_tracked_persons
                
                # Draw all tracked items on clean frame
                frame = frame_clean
                
                # Draw objects (yellow)
                for obj in tracked_objects:
                    if obj['age'] <= 2:  # Only current detections
                        x, y, w, h = obj['box']
                        label = f"{obj['label']} {obj['distance']:.0f}cm" if obj['distance'] else obj['label']
                        color = (0, 255, 255)  # Yellow
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw persons (green)
                for person in tracked_persons:
                    if person['age'] <= 2:  # Only current detections
                        x, y, w, h = person['box']
                        label = f"{person['name']} {person['distance']:.0f}cm" if person['distance'] else person['name']
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Announce detected items
                if current_time - last_announce_time > announce_interval:
                    current_items = []
                    announcements = []
                    
                    # Include all persons
                    for person in tracked_persons:
                        if person['age'] <= 2:
                            current_items.append({
                                'type': 'person',
                                'id': person['name'],
                                'position': person['position'],
                                'distance': person['distance']
                            })
                    
                    # Include ALL objects
                    for obj in tracked_objects:
                        if obj['age'] <= 2 and obj['confidence'] > 0.25:
                            current_items.append({
                                'type': 'object',
                                'id': obj['label'],
                                'position': obj['position'],
                                'distance': obj['distance']
                            })
                    
                    # Remove duplicates and generate announcements
                    seen_items = set()
                    for item in current_items:
                        item_key = f"{item['id']}-{item['position']}"
                        if item_key not in seen_items:
                            seen_items.add(item_key)
                            dist = f"{item['distance']:.0f} cm" if item['distance'] else "unknown distance"
                            announcements.append(f"{item['id']} {dist} {item['position']}")
                    
                    # Speak announcements if there are any
                    if announcements:
                        full_announcement = ". ".join(announcements)
                        registry.speak(full_announcement)
                        last_announced_items = announcements.copy()
                    
                    last_announce_time = current_time

        if registration_mode:
            cv2.putText(frame, "REGISTRATION MODE - Look at Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = registry.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(80,80))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            spoken_name = listen_for_name()
            if spoken_name:
                if registry.register_person(frame, spoken_name):
                    registry.speak(f"Registration successful for {spoken_name}")
                else:
                    registry.speak("Registration failed.")
            else:
                registry.speak("Name not recognized.")
            registration_mode = False
        
        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (width-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face and Object Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            registration_mode = True
            registry.speak("Starting registration mode")
        elif key == ord('o'):
            object_detection_enabled = not object_detection_enabled
            status = "enabled" if object_detection_enabled else "disabled"
            registry.speak(f"Object detection {status}")

finally:
    print("[SYSTEM] Shutting down...")
    registry.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Exit complete")