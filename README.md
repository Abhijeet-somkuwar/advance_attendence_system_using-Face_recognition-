# advance_attendence_system_using-Face_recognition-
Face Recognition Attendance System
This project is a simple face recognition attendance system implemented using Python and OpenCV. The system captures video from the default camera, detects faces, matches them against a set of known faces, and marks attendance.

Dependencies
Python 3.x
OpenCV
face_recognition
pandas

Install the required dependencies using: pip install opencv-python face_recognition pandas

Getting Started
1. Clone the repository to your local machine: git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
2. Create a virtual environment (optional but recommended): python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
3. Install dependencies: pip install -r requirements.txt
4. Create a folder named images and place the images of individuals you want to recognize inside it. Ensure the filenames follow the format name.jpg or name.jpeg.
5. Run the main script: python face_recognition_attendance.py
Press 'q' to exit the application.

Configuration
images_folder_path: Path to the folder containing images of known individuals.
webcam_video_stream: Initialize the video stream (default is set to the default camera).
attendance.xlsx: Excel file to store attendance records.
Notes
The attendance is marked and saved in an Excel file (attendance.xlsx).
Adjust the number_of_times_to_upsample parameter in the code based on the performance of your system.
Feel free to customize the code to suit your specific use case.

License
This project is licensed under the MIT License - see the LICENSE file for details.
