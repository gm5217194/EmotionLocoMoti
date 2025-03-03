from flask import Flask, render_template
import threading
import capture

app = Flask(__name__)

# Global variable to store detected emotion
detected_emotion = "No face detected"

def run_emotion_detection():
    global detected_emotion
    while True:
        detected_emotion = capture.get_emotion()

# Run emotion detection in background
threading.Thread(target=run_emotion_detection, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html', emotion=detected_emotion)

if __name__ == '__main__':
    app.run(debug=True)

