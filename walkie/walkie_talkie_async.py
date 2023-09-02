# rtl_fm -f 467812500 -l 50 | play -t raw -r 24k -es -b 16 -c 1 -V1 -
# in gqrx good settings were narrow FM, filter width  5kHz, squelch -55 dB, max deviation 2.5 kHz

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from speech_to_text import SpeechToText
import wave

sampling_rate = 24000
f_carrier = 467812500

stt = SpeechToText()

def process_message(message, show_wave=True):


    # writing to wav should not be needed remove later
    ntrim = int(0.5 * sampling_rate)
    message = message[ntrim:-2*ntrim]

    if show_wave:
        plt.plot(message)
        plt.ylim([-5000, 5000])
        plt.show()
    
    filename = 'temp.wav'
    with wave.open(filename, 'wb') as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(sampling_rate)
        fp.setnframes(len(message))
        fp.writeframes(message)

    out_text = stt.process_wav(filename)
    print('\nMessage Received:')
    print(out_text)


async def read_with_timeout(process, timeout):
    try:
        data = await asyncio.wait_for(process.stdout.read(sampling_rate*10), timeout)
        return data
    except asyncio.TimeoutError:
        #print("Listening")
        return None

async def main():
    cmd = f'rtl_fm -f {f_carrier} -s 24k -l 50 -E deemp'
    cmd_list = cmd.split()
    process = await asyncio.create_subprocess_exec(
        *cmd_list,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    data = []
    try:
        while not process.stdout.at_eof():
            data_chunk = await read_with_timeout(process, 1.0)
            if data_chunk is None:  # No data due to timeout
                if len(data) > 0:
                    message = np.concatenate(data)
                    process_message(message)
                    data = []
                continue

            # Convert bytes to numpy ndarray of int16 type
            x = np.frombuffer(data_chunk, dtype=np.int16)
            data.append(x)
            print(f"Receiving: {x[:10]}...")

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Initiating clean shutdown...")

    finally:
        if process.returncode is None:  # Process is still running
            process.terminate()
            await process.wait()  # Wait for the process to terminate
        print("Shutdown complete.")

    # Ensure the subprocess finishes
    await process.communicate()

if __name__ == "__main__":
    asyncio.run(main())
