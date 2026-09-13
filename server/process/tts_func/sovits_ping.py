import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
import yaml

# Load YAML config
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)


def play_audio(path):
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished

def sovits_gen(in_text, output_wav_pth = "output.wav"):
    url = "http://127.0.0.1:9880/tts"

    payload = {
        "text": in_text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": char_config['sovits_ping_config']['ref_audio_path'],  # Make sure this path is valid
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang']
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.ConnectionError:
        print(f"GPT-SoVITS is not reachable at {url}. Voice output skipped.")
        return None
    except requests.RequestException as exc:
        print(f"GPT-SoVITS could not generate audio: {exc}. Voice output skipped.")
        return None

    if not response.content:
        print("GPT-SoVITS returned empty audio. Voice output skipped.")
        return None

    with open(output_wav_pth, "wb") as f:
        f.write(response.content)
    return output_wav_pth



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)

