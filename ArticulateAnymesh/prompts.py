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


def recognize_parts_from_image(image_paths, object_name, save_file):
    
    user_prompt = f"""
Your job is to assist the user to analyze the structure of an object. Specifically, the user will give you {len(image_paths)} images of an articulated object, taken from different viewpoints. Your job is to recognize the main parts of that object. You should give your answer in the following format:

```part_list
(1) part_name: name of the part; description: a brief description about the part, and how it moves
(2) part_name: name of the part; description: a brief description about the part, and how it moves
...

```

Remember:
(1) Do not answer anything not asked.
(2) Your answer should be purely based on the input image, do not imagine anything.
(3) If there are multiple parts with the same semantic, just add one part to the list. For example, if there are four wheels, just add one part whose name is wheel.

USER INPUT:
Object: {object_name}
    """

    system_prompt = """
You are a helpful assistant. You have a good understanding of the structure of articulated objects.
    """

    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)


def generate_articulation_tree(object_name, part_list, save_path):
    user_prompt = f"""
Your job is to assist the user to analyze the structure of an object. Specifically, the user will name an object and then give you the main parts of that object. You will have to group these parts into links and then give the joint type of these links. 

After providing your analysis, you should give your answer in the following format:

```articulation tree
parts:
(1) part_name: name of the recognized part;
...

links:
(1) link_name: name of the link;
...

joints:
(1) joint_name: name of the joint; joint_type: type of the joint; parent_link: name of the parent link; child_link: name of the child link;
...
```

For example:
```articulation tree
parts:
(1) part_name: Front windshield;
(2) part_name: Door;
(3) part_name: Headlight;
(4) part_name: Wheel;
(5) part_name: Side window;

links:
(1) link_name: Base;
(2) link_name: Door;
(3) link_name: Wheel;
(4) part_name: Side window;

joints:
(1) joint_name: wheel_base_joint; joint_type: continuous; parent_link: Base; child_link: Wheel;
(2) joint_name: door_base_joint; joint_type: revolute; parent_link: Base; child_link: Door;
(3) joint_name: window_door_joint; joint_type: prismatic; parent_link: Door; child_link: Side window;
```

Remember:
(1) Do not answer anything not asked
(2) Available joint types are: fixed, prismatic, revolute and continuous
(3) For joint_type, only answer one word (among the available types), do not answer anything else
(4) For every part that is not fixed, there must be a unique link for it
(5) For parts that are fixed, try to group as many as you can
(6) There must be a Base link that serves as the root link

USER INPUT:
Object: {object_name}
Parts:
{part_list}
    """

    system_prompt = """
You are a helpful assistant. You have a good understanding of the structure of articulated objects. You are very familiar with URDF format.
    """

    response = call_gpt4o(system_prompt, user_prompt, [])

    with open(f"{save_path}/articulation_tree.txt", 'w') as file:
        file.write(response)

def generate_articulation_tree_known_parts(object_name, part_list, save_path):
    user_prompt = f"""
Your job is to assist the user to analyze the structure of an object. Specifically, the user will name an object and then give you the main parts of that object. You will have to group these parts into links and then give the joint type of these links. 

After providing your analysis, you should give your answer in the following format:

```articulation tree
parts:
(1) part_name: name of the recognized part;
...

links:
(1) link_name: name of the link;
...

joints:
(1) joint_name: name of the joint; joint_type: type of the joint; parent_link: name of the parent link; child_link: name of the child link;
...
```

For example:
```articulation tree
parts:
(1) part_name: Front windshield;
(2) part_name: Door;
(3) part_name: Headlight;
(4) part_name: Wheel;
(5) part_name: Side window;

links:
(1) link_name: Base;
(2) link_name: Door;
(3) link_name: Wheel;
(4) part_name: Side window;

joints:
(1) joint_name: wheel_base_joint; joint_type: continuous; parent_link: Base; child_link: Wheel;
(2) joint_name: door_base_joint; joint_type: revolute; parent_link: Base; child_link: Door;
(3) joint_name: window_door_joint; joint_type: prismatic; parent_link: Door; child_link: Side window;

```

Remember:
(1) Do not answer anything not asked
(2) Available joint types are: fixed, prismatic, revolute and continuous
(3) For joint_type, only answer one word (among the available types), do not answer anything else



USER INPUT:
Object: {object_name}
Parts:
{part_list}
    """

    system_prompt = """
You are a helpful assistant. You have a good understanding of the structure of articulated objects. You are very familiar with URDF format.
    """

    response = call_gpt4o(system_prompt, user_prompt, [])

    with open(f"{save_path}/articulation_tree.txt", 'w') as file:
        file.write(response)

def select_correct_hinge_two_points(image_paths, object_name, part_name, save_file):

    system_prompt = f"""
You are an assistant with a deep understanding of the structure of objects. 
    """

    user_prompt = f"""
Your task is to answer some questions about the input image of an object. The input image is of a {object_name}. The image has some points marked, each with an numerical ID as a label. Please select the points that are on the rotation axis of the {part_name} of the {object_name}. 

First provide your analysis, and then give your answer in the following format:
```hinge points
selected IDs: ID of selected point1, ID of selected point2, ...
```

An example output would be:
# YOUR ANALYSIS HERE...
```hinge points
selected IDs: 1, 3
```

Another example output would be:
# YOUR ANALYSIS HERE...
```hinge points
selected IDs: 2, 6, 7, 11, 13
```

Remember: 
(1) Do not answer anything not asked for.
(2) Select two or more points.
    """

    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)


def select_correct_hinge_one_point(image_paths, object_name, part_name, save_file):

    system_prompt = f"""
You are a helpful assistant with a deep understanding of the structure of objects. 
"""

    user_prompt = f"""
Your task is to answer some questions about the input image of an object. The input image is of a {object_name}. The image has some points marked, each with an numerical ID as a label. Please select the point that is on the rotation axis of the {part_name} of the {object_name}. 

First provide your analysis, and then give your answer in the following format:
```hinge points
selected IDs: ID of the selected point
```

Remember: 
(1) Do not answer anything not asked for.
(2) Only select one point that is the most suitable.
    """

    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)

def select_translation_direction(image_paths, object_name, part_name, save_file):

    system_prompt = f"""
You are an assistant with a deep understanding of the structure of objects. 
    """

    user_prompt = f"""
Your task is to answer some questions about the input image of an object. The input image is of a {object_name}. The image has some arrows marked, each with a different color. Please select the arrow that indicate the translation direction of the {part_name} of the {object_name}. 

First provide your analysis, and then give your answer in the following format:
```hinge points
selected arrow: color of the arrow (red, yellow, blue or green)
```

Remember: 
(1) Do not answer anything not asked for.
(2) Select one or two arrows (only if the two arrows point in opposite directions).
    """

    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)

def select_prismatic_method(image_paths, object_name, part_name, save_file):

    system_prompt = f"""
You are a helpful assistant with a deep understanding of the structure of objects. 
    """

    user_prompt = f"""
Your task is to help users to determine the translation direction of some parts of a given object mesh using common sense. It is important to note that the object is represented by a mesh, so you only have access to the object's surface and no access to its inner structure.

Specifically, the user will provide you with an object and the part for which the translation direction needs to be predicted will be specified. You will have to decide whether the translation direction is outwards from/inwards towards the mesh, or along the surface of the mesh.

When a part moves outwards, you will see more of that part coming out from the object. When a part moves inwards, you will see portions of that part going into the object. When a part moves along the surface of an object, you still see the exact same part. 

For example, a pressing button can be pressed inwards (when you press it, the button goes into the object), the telescopic handle of a suitcase can be pulled outwards (when you pull it, the entire handle comes out of the suitcase), and a stick shift moves along the surface of a shift pattern (when you are shifting, the shift does not go into the transmission or out of the transmission).

First provide your analysis, and then give your answer in the following format:
```slider_info
choice: outward/inward or surface
```

USER INPUT:
Object: {object_name}
Part: {part_name}
    """


    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)

def select_revolute_method(image_paths, object_name, part_name, save_file):

    system_prompt = f"""
You are a helpful assistant with a deep understanding of the structure of objects.
    """

    user_prompt = f"""
Your task is to help the users determine the hinge position of some parts of a given object mesh using common sense. It is important to note that the object is represented by a mesh, so you only have access to the object's surface and no access to its inner structure. The term "hinge" here does not only refer to the mechanical structure of a hinge, but also has a broader meaning. For example, the connection between a cardboard box lid and the body of the box is also considered a hinge.

Specifically, the user will provide you with an object, and the part for which the hinge position needs to be predicted will be specified. You will have to decide whether (1) both ends of the hinge are positioned on the surface of the object or (2) only one end of the hinge is positioned near the surface of the object, and the other end is inside the object.

For example, the hinge of a door has both its ends positioned on the door frame, which is recognizable and falls into the first category. The hinge of a wheel has one end recognizable on the center of the wheel, and its other end hidden inside the car or under the car (normally an object mesh of a car will not have detailed mechanical structures), which falls into the second category.

First provide your analysis, and then give your answer in the following format:
```hinge_info
choice: (1) or (2)
```

USER INPUT:
Object: {object_name}
Part: {part_name}
    """

    response = call_gpt4o(system_prompt, user_prompt, image_paths)

    with open(save_file, 'w') as file:
        file.write(response)


