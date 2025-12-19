import os
import cv2
import numpy as np
import pickle
import time
import queue
import threading
import pyttsx3

class PersonRegistry:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.registry_file = "face_registry.dat"
        self.training_file = "face_recognizer.yml"
        self.training_data_dir = "training_data"
        
        self.label_ids = {}
        self.current_id = 0
        self.confidence_threshold = 75
        
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 180)
        self.audio_queue = queue.Queue(maxsize=1)
        self.audio_thread = threading.Thread(target=self._audio_player, daemon=True)
        self.audio_thread.start()
        
        os.makedirs(self.training_data_dir, exist_ok=True)
        self.load_registry()
        self._verify_training()

    def _audio_player(self):
        while True:
            message = self.audio_queue.get()
            if message is None:
                break
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Audio playback error: {e}")
            finally:
                self.audio_queue.task_done()

    def speak(self, message):
        try:
            if not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self.audio_queue.put(message)
        except Exception as e:
            print(f"Error queuing speech: {e}")

    def _verify_training(self):
        if os.path.exists(self.training_file):
            try:
                self.recognizer.read(self.training_file)
                print("✅ Model loaded successfully")
            except:
                print("⚠️ Model file corrupted, retraining...")
                os.remove(self.training_file)
                self.train_recognizer()
        elif self.label_ids:
            print("⚠️ Model file missing, training...")
            self.train_recognizer()

    def load_registry(self):
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'rb') as f:
                    data = pickle.load(f)
                    self.label_ids = data.get('label_ids', {})
                    self.current_id = data.get('current_id', 0)
                print(f"📁 Loaded registry with {len(self.label_ids)} people")
                return True
        except Exception as e:
            print(f"❌ Failed to load registry: {e}")
        return False

    def save_registry(self):
        try:
            with open(self.registry_file, 'wb') as f:
                pickle.dump({
                    'label_ids': self.label_ids,
                    'current_id': self.current_id
                }, f)
            return True
        except Exception as e:
            print(f"❌ Failed to save registry: {e}")
            return False

    def register_person(self, frame, name):
        try:
            if frame is None or frame.size == 0:
                self.speak("No image detected")
                return False
            # converts the input frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # use Haar Cascade to detect face coordiante in the image.
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                self.speak("No clear face detected. Face the camera directly.")
                return False
            #If mutliple faces are detected the largest one is detected.
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0 or w < 80 or h < 80:
                self.speak("Face too small or unclear. Move closer.")
                return False

            if name in self.label_ids:
                self.speak(f"Name {name} already exists. Use a different name.")
                return False

            self.label_ids[name] = self.current_id
            
            for i in range(15):
                timestamp = int(time.time()) + i
                img_path = f"{self.training_data_dir}/{name}_{self.current_id}_{timestamp}.jpg"
                cv2.imwrite(img_path, face_roi)
                time.sleep(0.1)

            self.current_id += 1
            if self.train_recognizer():
                self.save_registry()
                self.speak(f"Successfully registered {name}")
                return True
                
            self.speak("Registration failed during training")
            return False

        except Exception as e:
            print(f"Registration error: {e}")
            self.speak("Registration failed due to an error")
            return False

    def train_recognizer(self):
        try:
            faces = []
            labels = []
            
            if not os.path.exists(self.training_data_dir):
                print("⚠️ No training data directory")
                return False

            training_files = [f for f in os.listdir(self.training_data_dir) if f.endswith(".jpg")]
            
            if not training_files:
                print("⚠️ No training images found")
                return False

            for file in training_files:
                path = os.path.join(self.training_data_dir, file)
                name = file.split("_")[0]
                
                if name in self.label_ids:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (150, 150))
                        faces.append(img)
                        labels.append(self.label_ids[name])
                    else:
                        print(f"⚠️ Could not read image: {path}")

            if len(faces) > 0:
                self.recognizer.train(faces, np.array(labels))
                self.recognizer.save(self.training_file)
                print(f"🔄 Retrained with {len(faces)} face samples")
                return True
            
            print("⚠️ No valid training data")
            return False

        except Exception as e:
            print(f"❌ Training error: {e}")
            return False

    def recognize_person(self, frame):
        try:
            if frame is None or frame.size == 0:
                return None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                return None

            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (150, 150))

            if face_roi.size == 0:
                return None

            if not self.label_ids:
                return None
            # Predit the label & confidence score using trained LBPH model.
            label_id, confidence = self.recognizer.predict(face_roi)
            
            if confidence < self.confidence_threshold:
                # finds the matching person's name.
                for name, id_ in self.label_ids.items():
                    if id_ == label_id:
                        print(f"[DEBUG] Recognized {name} with confidence {confidence:.1f}")
                        return name
            else:
                print(f"[DEBUG] Confidence too low: {confidence:.1f}")
            return None

        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return None

    def cleanup(self):
        self.audio_queue.put(None)
        self.audio_thread.join()