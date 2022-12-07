import requests
import os
import sys
import io
from PIL import Image
import json
import base64

#
# You can generate your Tiyaro API Key from here
# https://console.tiyaro.ai/apikeys
#
apikey = os.getenv("TIYARO_API_KEY")
if apikey is None:
    print("Please set env variable TIYARO_API_KEY")
    print("You can generate your API key from https://console.tiyaro.ai/apikeys")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {apikey}",
    "content-type": "application/json"
}
url = "https://api.tiyaro.ai/v1/ent/huggingface/1/stabilityai/stable-diffusion-2-base"
querystring = {"serviceTier": "gpuflex"}

#
# Credit for the prompt goes to this reddit post
# https://www.reddit.com/r/StableDiffusion/comments/z4r2oo/v2_makes_really_nice_birds/?utm_source=share&utm_medium=web2x&context=3
#
payload = {"input": {
    "prompt": "Cinematic shot of taxidermy bird inside an antique shop, glass, crystal, flowers, loxurious,",
    "negative_prompt": "Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render,",
    "scheduler": "dpm",
    "disable_safety_checker": False,
    "seed": 3813511178,
    "steps": 20,
    "guidance_scale": 12.1,
    "width": 512,
    "height": 512
}}

response = requests.request(
    "POST", url, json=payload, headers=headers, params=querystring)

resp = json.loads(response.text)
imgB64 = resp["response"]["images"][0]
imgData = base64.b64decode(imgB64)

im = Image.open(io.BytesIO(imgData))
im.show()
