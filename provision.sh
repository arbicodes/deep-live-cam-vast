#!/bin/bash
set -e

echo "========================================="
echo "Deep Live Cam - Automated Setup"
echo "========================================="

# Step 1: Clone Repository
echo "[1/9] Cloning Deep-Live-Cam repository..."
cd /workspace
if [ ! -d "Deep-Live-Cam" ]; then
    git clone https://github.com/hacksider/Deep-Live-Cam.git
fi
cd Deep-Live-Cam

# Step 2: Download Models
echo "[2/9] Downloading AI models (this may take a few minutes)..."
mkdir -p models
cd models

if [ ! -f "inswapper_128.onnx" ]; then
    echo "  - Downloading inswapper_128.onnx (529MB)..."
    curl -L -o inswapper_128.onnx 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx'
fi

if [ ! -f "inswapper_128_fp16.onnx" ]; then
    echo "  - Downloading inswapper_128_fp16.onnx (265MB)..."
    curl -L -o inswapper_128_fp16.onnx 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'
fi

if [ ! -f "GFPGANv1.4.pth" ]; then
    echo "  - Downloading GFPGANv1.4.pth (348MB)..."
    curl -L -o GFPGANv1.4.pth 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth'
fi

# Step 3: Verify Model Files
echo "[3/9] Verifying model files..."
ls -lh /workspace/Deep-Live-Cam/models/
echo ""
echo "Model file sizes:"
du -h /workspace/Deep-Live-Cam/models/*

# Step 4: Install Dependencies
echo "[4/9] Installing Python dependencies..."
cd /workspace/Deep-Live-Cam

# Try full install first
pip install -r requirements.txt 2>/dev/null || true

# Install essential packages with --ignore-installed flag
echo "  - Installing essential packages..."
pip install --ignore-installed flask flask-cors websockets insightface onnxruntime-gpu opencv-python numpy

# Step 5: Verify CUDA Setup
echo "[5/9] Verifying CUDA setup..."
python3 -c "import onnxruntime; print('ONNXRuntime:', onnxruntime.__version__); print('Providers:', onnxruntime.get_available_providers())"

# Step 6: Test Face Detection
echo "[6/9] Testing face detection..."
python3 -c "
import cv2
from insightface.app import FaceAnalysis

print('Loading face analyzer...')
fa = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(640, 640))
print('✓ Face analyzer loaded successfully!')
"

# Step 7: Create Web Server
echo "[7/9] Creating web server..."
cat > /workspace/Deep-Live-Cam/web_server.py << 'WEBSERVER_EOF'
from flask import Flask, render_template, request, Response, jsonify, session
import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
import threading
import os
import uuid

app = Flask(__name__)
app.secret_key = 'deeplivecam-secret-key-change-me'

# Will be set to Cloudflare URLs via environment variables
WEB_URL = os.environ.get('WEB_URL', 'http://localhost:5000')
WS_URL = os.environ.get('WS_URL', 'ws://localhost:8888')

# Global state
face_analyzer = None
face_swapper = None

# Per-user source faces
user_faces = {}
user_settings = {}

def init_models():
    global face_analyzer, face_swapper
    print("Loading face analyzer...")
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Loading face swapper...")
    face_swapper = insightface.model_zoo.get_model('/workspace/Deep-Live-Cam/models/inswapper_128.onnx', providers=['CUDAExecutionProvider'])
    print("Models loaded!")

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())[:8]
    user_id = session['user_id']

    if user_id not in user_settings:
        user_settings[user_id] = {'mouth_mask': False, 'many_faces': False}

    print(f"User connected: {user_id}")

    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Deep Live Cam</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d9ff; text-align: center; }
        .user-id { text-align: center; font-size: 12px; color: #666; margin-bottom: 10px; }
        .panel { display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }
        .video-box { background: #16213e; padding: 15px; border-radius: 10px; position: relative; }
        video, #output { width: 640px; height: 480px; background: #000; border-radius: 5px; }
        .controls { background: #16213e; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .control-row { display: flex; align-items: center; gap: 15px; margin: 10px 0; }
        label { min-width: 120px; }
        input[type="file"] { background: #0f3460; padding: 10px; border-radius: 5px; color: white; }
        button { background: #00d9ff; color: #1a1a2e; border: none; padding: 12px 25px; border-radius: 5px; cursor: pointer; font-weight: bold; }
        button:hover { background: #00b8d4; }
        button:disabled { background: #555; cursor: not-allowed; }
        .toggle { width: 50px; height: 26px; background: #555; border-radius: 13px; position: relative; cursor: pointer; }
        .toggle.active { background: #00d9ff; }
        .toggle::after { content: ''; width: 22px; height: 22px; background: white; border-radius: 50%; position: absolute; top: 2px; left: 2px; transition: 0.2s; }
        .toggle.active::after { left: 26px; }
        #status { text-align: center; padding: 10px; color: #00d9ff; }
        #sourcePreview { max-width: 150px; max-height: 150px; border-radius: 5px; margin-top: 10px; }
        .btn-group { position: absolute; top: 25px; right: 25px; display: flex; gap: 10px; }
        .output-btn { background: rgba(0,217,255,0.8); border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; font-size: 14px; }
        .output-btn:hover { background: #00d9ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Deep Live Cam</h1>
        <div class="user-id">Session: ''' + user_id + '''</div>
        <div id="status">Upload a source face to begin</div>

        <div class="panel">
            <div class="video-box">
                <h3>📷 Your Camera</h3>
                <video id="webcam" autoplay playsinline muted></video>
            </div>
            <div class="video-box">
                <h3>🎭 Output</h3>
                <div class="btn-group">
                    <button class="output-btn" onclick="popOut()">🪟 Pop Out</button>
                    <button class="output-btn" onclick="goFullscreen()">⛶ Fullscreen</button>
                </div>
                <canvas id="output" width="640" height="480"></canvas>
            </div>
        </div>

        <div class="controls">
            <div class="control-row">
                <label>Source Face:</label>
                <input type="file" id="sourceFile" accept="image/*">
                <img id="sourcePreview" style="display:none;">
            </div>
            <div class="control-row">
                <label>Many Faces:</label>
                <div class="toggle" id="manyFaces" onclick="toggleSetting(this, 'many_faces')"></div>
            </div>
            <div class="control-row">
                <label>Mouth Mask:</label>
                <div class="toggle" id="mouthMask" onclick="toggleSetting(this, 'mouth_mask')"></div>
            </div>
            <div class="control-row">
                <button id="startBtn" onclick="startStream()" disabled>▶️ Start</button>
                <button id="stopBtn" onclick="stopStream()" disabled>⏹️ Stop</button>
            </div>
        </div>
    </div>

    <script>
        const USER_ID = "''' + user_id + '''";
        const WS_URL = "''' + WS_URL + '''";
        let streaming = false;
        let ws = null;
        let pendingFrame = false;
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('output');
        const ctx = canvas.getContext('2d');

        function popOut() {
            window.open('/popout', 'DeepLiveCam', 'width=660,height=520,menubar=no,toolbar=no,location=no,status=no');
        }

        function goFullscreen() {
            if (canvas.requestFullscreen) {
                canvas.requestFullscreen();
            } else if (canvas.webkitRequestFullscreen) {
                canvas.webkitRequestFullscreen();
            }
        }

        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then(stream => { video.srcObject = stream; })
            .catch(err => console.error('Camera error:', err));

        document.getElementById('sourceFile').onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('status').textContent = 'Uploading...';

            if (streaming) {
                stopStream();
                await new Promise(r => setTimeout(r, 500));
            }

            const resp = await fetch('/upload_source', { method: 'POST', body: formData });
            const data = await resp.json();

            if (data.success) {
                document.getElementById('status').textContent = 'Source face loaded! Click Start to begin.';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('sourcePreview').src = URL.createObjectURL(file);
                document.getElementById('sourcePreview').style.display = 'block';
            } else {
                document.getElementById('status').textContent = 'Error: ' + data.error;
            }
        };

        function toggleSetting(el, setting) {
            el.classList.toggle('active');
            fetch('/setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: setting, value: el.classList.contains('active')})
            });
        }

        function startStream() {
            streaming = true;
            pendingFrame = false;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('status').textContent = 'Connecting...';

            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                ws.send(JSON.stringify({type: 'auth', user_id: USER_ID}));
                document.getElementById('status').textContent = 'Streaming...';
                sendFrame();
            };

            ws.onmessage = (event) => {
                pendingFrame = false;
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    if (window.popupCanvas) {
                        window.popupCanvas.getContext('2d').drawImage(img, 0, 0, 640, 480);
                    }
                    if (streaming) {
                        requestAnimationFrame(sendFrame);
                    }
                };
                img.src = 'data:image/jpeg;base64,' + event.data;
            };

            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
                document.getElementById('status').textContent = 'Connection error - check console';
            };

            ws.onclose = () => {
                if (streaming) {
                    document.getElementById('status').textContent = 'Disconnected. Click Start to reconnect.';
                    streaming = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                }
            };
        }

        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN || pendingFrame) return;

            pendingFrame = true;
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 640;
            tempCanvas.height = 640;
            tempCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            const data = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];

            try {
                ws.send(data);
            } catch (e) {
                console.error('Send error:', e);
                pendingFrame = false;
            }
        }

        function stopStream() {
            streaming = false;
            pendingFrame = false;
            if (ws) {
                ws.close();
                ws = null;
            }
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').textContent = 'Stopped';
        }
    </script>
</body>
</html>
'''

@app.route('/popout')
def popout():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Deep Live Cam Output</title>
    <style>
        * { margin: 0; padding: 0; }
        body { background: #000; overflow: hidden; }
        canvas { display: block; width: 100vw; height: 100vh; object-fit: contain; }
    </style>
</head>
<body>
    <canvas id="popupOutput" width="640" height="480"></canvas>
    <script>
        const canvas = document.getElementById('popupOutput');
        if (window.opener) {
            window.opener.popupCanvas = canvas;
        }
        window.onbeforeunload = () => {
            if (window.opener) {
                window.opener.popupCanvas = null;
            }
        };
    </script>
</body>
</html>
'''

@app.route('/upload_source', methods=['POST'])
def upload_source():
    user_id = session.get('user_id', 'default')
    try:
        file = request.files['file']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = face_analyzer.get(img)
        if not faces:
            return jsonify({'success': False, 'error': 'No face detected in image'})

        user_faces[user_id] = faces[0]
        print(f"New source face loaded for user {user_id}!")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/setting', methods=['POST'])
def update_setting():
    user_id = session.get('user_id', 'default')
    if user_id not in user_settings:
        user_settings[user_id] = {'mouth_mask': False, 'many_faces': False}

    data = request.json
    user_settings[user_id][data['setting']] = data['value']
    return jsonify({'success': True})

import asyncio
import websockets
import json

async def process_frame(websocket):
    user_id = None
    print("Client connecting...")

    try:
        async for message in websocket:
            try:
                if message.startswith('{'):
                    data = json.loads(message)
                    if data.get('type') == 'auth':
                        user_id = data.get('user_id', 'default')
                        print(f"User {user_id} authenticated")
                        continue

                if not user_id:
                    user_id = 'default'

                img_data = base64.b64decode(message)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                source_face = user_faces.get(user_id)
                settings = user_settings.get(user_id, {'many_faces': False})

                if frame is not None and source_face is not None:
                    faces = face_analyzer.get(frame)
                    targets = faces if settings.get('many_faces') else faces[:1]
                    for target_face in targets:
                        frame = face_swapper.get(frame, target_face, source_face, paste_back=True)

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await websocket.send(base64.b64encode(buffer).decode())
            except Exception as e:
                print(f"Frame error for user {user_id}: {e}")
                if 'frame' in dir() and frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    await websocket.send(base64.b64encode(buffer).decode())
    except websockets.exceptions.ConnectionClosed:
        print(f"User {user_id} disconnected")
    except Exception as e:
        print(f"Connection error: {e}")

async def ws_server():
    print("Starting WebSocket server on port 8888...")
    async with websockets.serve(process_frame, "0.0.0.0", 8888, max_size=10*1024*1024):
        await asyncio.Future()

def run_ws():
    asyncio.run(ws_server())

if __name__ == '__main__':
    init_models()
    threading.Thread(target=run_ws, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
WEBSERVER_EOF

# Step 8: Kill Any Existing Processes on Ports
echo "[8/9] Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:8888 | xargs kill -9 2>/dev/null || true
pkill -f web_server.py 2>/dev/null || true

# Step 9: Create helper script
echo "[9/9] Creating helper script..."
cat > /workspace/get-link.sh << 'HELPER_EOF'
#!/bin/bash

echo "========================================="
echo "Deep Live Cam - Get Public Link"
echo "========================================="
echo ""

# Start the server if not running
if ! pgrep -f web_server.py > /dev/null; then
    echo "Starting server..."
    cd /workspace/Deep-Live-Cam
    nohup python3 web_server.py > server.log 2>&1 &
    echo $! > server.pid
    sleep 15
    echo "✓ Server started"
    echo ""
fi

# Check if server is running
if ! pgrep -f web_server.py > /dev/null; then
    echo "❌ ERROR: Server failed to start"
    echo "Check logs: tail -50 /workspace/Deep-Live-Cam/server.log"
    exit 1
fi

echo "✓ Server is running!"
echo ""
echo "To get your public link, add these ports in Vast.ai Instance Portal:"
echo ""
echo "1. Click the 'Open' button on your instance"
echo "2. In the Instance Portal, add two Cloudflare tunnels:"
echo "   - Port 5000 (HTTP - Web Interface)"
echo "   - Port 8888 (WebSocket - Video Streaming)"
echo ""
echo "3. The Instance Portal will give you two HTTPS URLs"
echo "4. Visit the port 5000 URL in your browser to use Deep Live Cam!"
echo ""
echo "Note: You can share the port 5000 URL with friends!"
echo ""
echo "Server logs: tail -f /workspace/Deep-Live-Cam/server.log"
echo "========================================="
HELPER_EOF

chmod +x /workspace/get-link.sh

# Start the server
echo ""
echo "========================================="
echo "Starting Deep Live Cam server..."
echo "========================================="
cd /workspace/Deep-Live-Cam
nohup python3 web_server.py > server.log 2>&1 &
echo $! > server.pid
sleep 15

# Verify server started
if pgrep -f web_server.py > /dev/null; then
    echo ""
    echo "✅✅✅ SETUP COMPLETE! ✅✅✅"
    echo ""
    echo "To get your public link, run this command:"
    echo ""
    echo "    /workspace/get-link.sh"
    echo ""
    echo "Or use the Instance Portal 'Open' button to add Cloudflare tunnels!"
    echo ""
else
    echo ""
    echo "❌ Server failed to start. Check logs:"
    echo "tail -50 /workspace/Deep-Live-Cam/server.log"
fi
