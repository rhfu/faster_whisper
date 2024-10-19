import asyncio
import os
import wave
import tempfile
import torch
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper
import websockets
import json

# Audio buffer time
AUDIO_BUFFER = 5

# Dictionary to store WebSocket connections
clients = {}

# handle client
async def handle_client(websocket):
   client_id = id(websocket)  # Using the WebSocket object's ID as a unique identifier
   print(f"Client connected from {websocket.remote_address} with ID {client_id}")
   clients[client_id] = websocket
   try:
      # print(f"Client connected from {websocket.remote_address}")
      # Wait for messages from the client
      async for message in websocket:
         print(f"Received message from client {client_id}: {message}")
         # Process the message (you can replace this with your logic)
         response = f"Server received: {message}"
         # Send a response back to the client
         await websocket.send(response)
         print(f"Sent response to client {client_id}: {response}")

   except websockets.exceptions.ConnectionClosedError:
      print(f"Connection with {websocket.remote_address} closed.")
   finally:
      # Remove the WebSocket connection when the client disconnects
      del clients[client_id]

# Send a message to all connected clients
async def send_all_clients(message):
   if clients==None or clients=={}: 
      print("No clients connected.")
      return
   for client_id, websocket in clients.items():
      try:
         await websocket.send(message)
         print(f"Sent message to client {client_id}: {message}")
      except Exception as e:
         print(f"Error sending message to client {client_id}: {e}")

# Send a message to a specific client identified by client_id
async def send_message(client_id, message):
   if client_id in clients:
      websocket = clients[client_id]
      await websocket.send(message)
      print(f"Sent message to client {client_id}: {message}")
   else:
      print(f"Client with ID {client_id} not found.")


# Start the server
async def main_server():
   server = await websockets.serve(handle_client, "localhost", 8765)
   print("WebSocket server started. Listening on ws://localhost:8765")

   await server.wait_closed()



#This function records audio using the PyAudio library and saves it as a temporary WAV file.
#Use pyaudio PyAudio instance creates an audio stream and writes audio data in real-time to a WAV file by specifying the callback function callback.
#Due to the use of the asyncio library, it is no longer necessary to use time. sleep() to block execution, but instead to use asyncio. sleep() to wait asynchronously.
#Finally, the function returns the file name of the saved WAV file.
async def record_audio(p, device):
    """Record audio from output device and save to temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(device["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(device["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=device["maxInputChannels"],
            rate=int(device["defaultSampleRate"]),
            frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt16),
            input=True,
            input_device_index=device["index"],
            stream_callback=callback,
        )

        await asyncio.sleep(AUDIO_BUFFER)

        stream.stop_stream()
        stream.close()
        wave_file.close()
        # print(f"{filename} saved.")
    return filename
 
# SegmentData class
class SegmentData:
    def __init__(self, start, end,text):
        # 实例属性
        self.start = start
        self.end = end
        self.text = text

    def __dict__(self):
        return {"start": self.start, "end": self.end, "text": self.text}

def convert_to_unity_data(data):  # 参数 data 为字典列表
    unity_data = []
    for item in data:
        segment_data = SegmentData(item["start"], item["end"], item["text"])
        unity_data.append(segment_data)
    return unity_data


# This function transcribes the recorded audio using the Whisper model and outputs the transcription result.
async def whisper_audio(filename, model):
    """Transcribe audio buffer and display."""
    segments, info = model.transcribe(filename, beam_size=5, language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    os.remove(filename)
    # print(f"{filename} removed.")
    if segments:
        segments_dict_list = [{"start": segment.start, "end": segment.end, "text": segment.text.strip()} for segment in segments]
        json_transcriptions=json.dumps(segments_dict_list)
        print(f"Transcription: {json_transcriptions}")
        try:
            await send_all_clients(json_transcriptions)
        except Exception as e:
            print(f"Error sending message: {e}")



# Start recording audio using PyAudio and concurrently run the whisper_audio function for audio transcription using asyncio.gather.
async def main():
    """Load model record audio and transcribe from default output device."""
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    model = whisper("large-v3", device=device, local_files_only=True,compute_type="int8_float16")

    print("Model loaded.")

    with pyaudio.PyAudio() as pya:
        # Get microphone device information (assuming you want to select the first microphone device)
        microphone_index = 0
        microphone_info = pya.get_device_info_by_index(microphone_index)
        while True:
            filename = await record_audio(pya, microphone_info)
            await asyncio.gather(whisper_audio(filename, model))


async def appmain():
    await asyncio.gather(main(), main_server())  # Gather coroutines here

if __name__ == "__main__":
    asyncio.run(appmain())  # Pass the main coroutine to asyncio.run()
