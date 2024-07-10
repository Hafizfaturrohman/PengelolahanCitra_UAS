# Nama Kelompok 
1. **Nama:** [HAFIZ FATURROHMAN]
   - **NIM:** [312210375]
   - **Kelas:** [TI.22.A.4]

2. **Nama:** [YUDHA EKA PERDANA]
   - **NIM:** [312210362]
   - **Kelas:** [TI.22.A.4]

3. **Nama:** [AZZAM SAUQI RABBANI]
   - **NIM:** [312210373]
   - **Kelas:** [TI.22.A.4]

4. **Nama:** [MUHAMMAD ARIFIN]
   - **NIM:** [312210330]
   - **Kelas:** [TI.22.A.4]
   
- Mata Kuliah:	Pengolahan Citra	
- Dosen Pengampu:	Muhammad Fatchan, S.Kom., M.Kom., MTCNA.

# Tampilan
https://github.com/Hafizfaturrohman/PengolahanCitra_UAS/assets/115616365/3d9158a8-96c1-4771-ae23-2c4e0bd4af71

https://github.com/Hafizfaturrohman/PengolahanCitra_UAS/assets/115616365/749d7872-37fc-49a8-9b82-6458743d9216

# Requirements
- streamlit==1.36.0
- numpy
- matplotlib
- opencv-python
- Pillow

# Installation
![Screenshot 2024-07-09 130807](https://github.com/Hafizfaturrohman/PengolahanCitra_UAS/assets/115616365/07fe7311-0b8b-4fe5-a094-25b100f13a0e)

# Project UTS
Simpan kode berikut dalam file bernama app.py
```import streamlit as st
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def main():
    st.title("Face and Eye Detection App")
    st.write("This app detects faces and eyes using Haar Cascades in real-time from your laptop camera or uploaded photos.")

    option = st.selectbox("Choose input source", ("Laptop Camera", "Upload Photo"))

    if option == "Laptop Camera":
        start_button = st.button("Start Camera")
        if start_button:
            run_camera_detection()

    elif option == "Upload Photo":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            num_faces = len(faces)
            message_faces = f"Detected {num_faces} face(s) in the image."
            st.write(message_faces)

            for (x, y, w, h) in faces:
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for faces
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image_np[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green rectangle for eyes

            st.image(image_np, channels="BGR")

def run_camera_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for faces
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green rectangle for eyes

        cv2.imshow('Face and Eye Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()```
    
# hasil UTS
![Screenshot 2024-07-10 213622](https://github.com/Hafizfaturrohman/PengolahanCitra_UAS/assets/115616365/0452e5ed-f603-4ac8-a822-725c49271da7)

![Screenshot 2024-07-10 214412](https://github.com/Hafizfaturrohman/PengolahanCitra_UAS/assets/115616365/0314e404-278c-43b3-b77b-664a8b648f26)


