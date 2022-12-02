import requests
import json
import os
import sys
import time

PROD_BASE = 'https://api.tiyaro.ai'
WHISPER_LARGE_ASYNC_PATH = f'{PROD_BASE}/v1/async/ent/tiyarofs/1/openai/whisper-large?serviceTier=gpuflex'
MP3_UPLOAD_URL = f'{PROD_BASE}/v1/input/upload-url?extension=mp3'


def getHeaders():
    api_key = os.environ.get('TIYARO_API_KEY')
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }


def whisper_input():
    return {
        "no_speech_threshold": 0.6,
        "patience": 1,
        "suppress_tokens": "-1",
        "compression_ratio_threshold": 2.4,
        #
        # NOTE Remove 'language' parame if you want native language
        #
        "language": "en",
        "temperature_increment_on_fallback": 0.2,
        "length_penalty": None,
        "logprob_threshold": -1,
        "condition_on_previous_text": True,
        "initial_prompt": None,
        "task": "transcribe",
        "temperature": 0,
        "beam_size": 5,
        "best_of": 5
    }


def get_upload_url(extension='mp3'):
    resp = requests.request("POST", MP3_UPLOAD_URL,
                            json={}, headers=getHeaders())
    assert resp.status_code == 201
    result = json.loads(resp.text)
    uploadURL = result['uploadUrl']['PUT']
    print('-- Input payload_url --', uploadURL)
    return uploadURL


def upload_mp3_to_url(mp3File, upload_url):
    resp = requests.request("PUT", upload_url, data=open(mp3File, 'rb'))
    assert resp.status_code == 200
    print(f'-- {mp3File} uploaded --')


def send_async_infer_request(upload_url):
    modelURL = WHISPER_LARGE_ASYNC_PATH

    payload = {
        "input": whisper_input(),
        "URL": upload_url
    }

    resp = requests.post(modelURL, headers=getHeaders(), json=payload)
    assert resp.status_code == 202
    print('-- async request submitted --')
    result = json.loads(resp.text)
    request_id = result['response']['id']
    print(f'requestId: {request_id}')
    return result['response']['urls']['GET']


def check_status_and_result(inference_result_url):
    status = "NA"
    result = None
    while True:
        resp = requests.request(
            "GET", inference_result_url, headers=getHeaders())
        assert resp.status_code == 200
        result = json.loads(resp.text)
        status = result["status"]
        if status == 'success':
            print("status: ", status)
            break
        print("status: ", status)
        time.sleep(15)
    print(json.dumps(result, indent=2))
    text = result["result"]["text"]
    print("-- Transcribed Text --\n", text)
    print("-- Done -- \n")


def async_infer(input_mp3):
    # Step 1 - Get a presigned url to upload your audio file
    upload_url = get_upload_url()

    # Step 2 - Upload your mp3 file to the presinged url
    upload_mp3_to_url(input_mp3, upload_url)

    # Step 3 - Submit an Async request. You get a inference_result_url that you can poll on.
    inference_result_url = send_async_infer_request(upload_url)

    # Step 4 - Poll/Wait for request to finish
    check_status_and_result(inference_result_url)


def main():
    api_key = os.environ.get('TIYARO_API_KEY')
    if not api_key:
        raise ValueError("TIYARO_API_KEY not set")

    if len(sys.argv) != 2:
        print(" Usage: python asyncWhisper.py <mp3_file>\n\n",
              "You can use the sample.mp3 included in this directory.\n",
              "Usage: python asyncWhisper.py sample.mp3\n")
        sys.exit(1)

    input_mp3 = sys.argv[1]
    print("-- processing input file --", input_mp3)

    start = time.time()
    async_infer(input_mp3)

    print("\n--- Inference time:", round(time.time() - start, 2), "secs ---")


if __name__ == "__main__":
    main()
