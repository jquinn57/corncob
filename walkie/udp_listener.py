import socket
import numpy as np
import threading
import matplotlib.pyplot as plt
import queue
from speech_to_text import SpeechToText
import wave

# broadcast to UDP using GQRX then listen with:
# nc -l -u 7355 | play -r 48k -b 16 -es -t raw -c 1 -V1 -
# or  vlc --demux=rawaud --rawaud-channels=1 --rawaud-samplerate=48000 udp://@:7355
# the vlc method is not choppy

# Settings
SAMPLING_RATE = 48000
MAX_LEN = SAMPLING_RATE * 10
PORT = 7355
BUFFER_SIZE = 24000
THRESHOLD = 1000  # TODO: adjust
running = True

def process_stt(message, stt):
    # writing to wav should not be needed remove later
    filename = 'temp.wav'
    with wave.open(filename, 'wb') as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(SAMPLING_RATE)
        fp.setnframes(len(message))
        fp.writeframes(message)
    out_text = stt.process_wav(filename)
    print('\nMessage Received:')
    print(out_text)
    print()

def process_signal(message_queue):
    global running
    stt = SpeechToText()
    while running:
        try:
            message = message_queue.get(timeout=1)
        except queue.Empty:
            message = None
        if message is not None:
            #plt.plot(message)
            #plt.show()
            process_stt(np.array(message, dtype=np.int16), stt)


message_queue = queue.Queue()
processing_thread = threading.Thread(target=process_signal, args=(message_queue,))
processing_thread.start()

def main():
    global running
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', PORT))
    print(f'Listining to UDP on port: {PORT}')
    
    below_threshold_count = 0
    current_signal = []

    while running:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            
            # Convert bytes to numpy ndarray of int16 type
            x = np.frombuffer(data, dtype=np.int16)

            # Check if signal is below threshold
            if np.mean(np.abs(x)) < THRESHOLD:
                below_threshold_count += 1
            else:
                below_threshold_count = 0

                # If we're not already collecting a signal, start now
                if not current_signal:
                    print("Starting capture")
                current_signal.extend(x.tolist())

            # If signal stays below threshold for 1 second
            # Pass the signal for processing and reset
            if below_threshold_count > (SAMPLING_RATE / BUFFER_SIZE) or len(current_signal) > MAX_LEN:
                if current_signal:
                    print("End capture, queue for processing")
                    message_queue.put(current_signal)
                    current_signal = []
        except KeyboardInterrupt:
            print('shuting down')
            sock.close()
            running = False
            processing_thread.join()

if __name__ == '__main__':
    main()
