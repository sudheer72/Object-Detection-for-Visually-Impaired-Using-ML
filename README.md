## Object Detection for Visually Impaired Using ML

## 📌 Overview
This project is a real-time, low-cost assistive system designed to help visually impaired individuals by providing audio-based awareness of their surroundings. It detects objects, recognizes familiar faces, estimates distance, and delivers voice feedback, enabling safer and more independent navigation.

The system is optimized to run **offline on CPU-only systems**, making it suitable for low-resource environments without internet or GPU dependency.

## ✨ Features
- Real-time object detection using **YOLOv3-tiny** trained on the **COCO dataset**
- Face detection using **Haar Cascade Classifier**
- Face recognition and user registration using **LBPH**
- Voice-based feedback and alerts using **pyttsx3**
- Voice-based user registration (hands-free interaction)
- Distance estimation using the **pinhole camera model**
- Offline, lightweight, and CPU-efficient execution

## 🛠️ Technologies Used
- Python  
- OpenCV  
- YOLOv3-tiny (COCO Models)  
- Haar Cascade  
- LBPH Face Recognizer  
- TensorFlow  
- pyttsx3 (Text-to-Speech)  
- Google Speech Recognition API / CMU Sphinx (offline fallback)

## ▶️ How It Works
1. Camera captures real-time video frames  
2. YOLOv3-tiny detects objects in the scene  
3. Haar Cascade detects faces and LBPH recognizes registered users  
4. Distance to detected objects is estimated using bounding box dimensions  
5. Detected objects, faces, and distances are converted into speech  
6. Audio feedback is delivered to the user in real time  

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/sudheer72/Object-Detection-for-Visually-Impaired-Using-ML.git
````

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the main Python file:

   ```bash
   python main.py
   ```


