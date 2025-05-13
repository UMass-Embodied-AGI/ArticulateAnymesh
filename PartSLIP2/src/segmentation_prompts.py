import os
import json
import requests
import base64
from openai import OpenAI

import PIL.Image

GPT_ENDPOINT = os.environ['GPT_ENDPOINT']
API_KEY = os.environ['API_KEY']

def call_gpt4o(system_prompt, user_prompt, image_paths):
     
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    for image_path in image_paths:
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }

        payload["messages"][1]["content"].append(image_content)

    try:
        response = requests.post(GPT_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    response = response.json()['choices'][0]['message']['content']

    return response.lower()


def multiview_parts_recognition(image_paths, object_name, num_views, parts_list, save_path):
  user_prompt = f"""
The user will provide you with {num_views} images, which are pictures of the same {object_name} taken from different viewpoints. You need to carefully examine these images and analyze which parts appear in which images.

The parts we care about are:
{", ".join(parts_list)}

First analyze the orientation of the {object_name} in each given image, then recognize the parts in these images. Only after you finished providing the analysis, provide your answer in the following format: 

```part_recognition
  part_name: name of the part; views: indices of images that have this part appearing in them, separated by commas
  part_name: name of the part; views: indices of images that have this part appearing in them, separated by commas
  ...

```

  Remember, image indices start with 0.
  """

  system_prompt = "You are a helpful assistant."

  response = call_gpt4o(system_prompt, user_prompt, image_paths)

  with open(f"{save_path}/multiview_recognition.txt", 'w') as file:
    file.write(response)

def merge_parts(image_paths, object_name, parts_list, save_path):

  user_prompt = f"""
The user will provide you with an image of {object_name}. The object in the image is divided into smaller segments, each lablled with a segmentation mask and a number on the mask. For each semantic part of the object, please choose the segments that belong to the semantic part. 
First give an analysis of the semantics of each segment, then provide your answer in the following format:

```part_list
1. name: name of the part; labels: the ID of segments that belong to this component, separated by comma
2. name: name of the part; labels: the ID of segments that belong to this component, separated by comma
...
```

The semantic parts we care about are:
{", ".join(parts_list)}

Remember: 
(1) We only care about the listed parts, do not add anything else to the part list
(2) If some of the listed semantic parts do not exist in the image, omit these semantic parts
(3) If there are multiple instances of a semantic part in the image, use two semantic parts. For example, if a fridge has two doors, use semantic part door@1 and door@2 (the format is semantic_part_name@unique_id).
  """

  system_prompt = "You are a helpful assistant."

  response = call_gpt4o(system_prompt, user_prompt, image_paths)

  with open(f"{save_path}/masks_to_links.txt", 'w') as file:
    file.write(response)
