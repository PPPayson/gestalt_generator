import os
import json
import base64
import mimetypes
import argparse

from ollama import chat, show
from openai import OpenAI
from pydantic import BaseModel

from typing import Literal
from pydantic import BaseModel, Field



class ImageDescription(BaseModel):
    n_objects: int = Field(
        ge=0,
        le=20,
        description="Total number of visually distinct objects in the image."
    )

    visible_shapes: list[str] = Field(
        min_length=0,
        max_length=20,
        description=(
            "All visually distinct shapes perceived in the image. "
            "Count repeated shapes separately. "
        )
    )

class IllusionResponse(BaseModel):
    has_illusion: bool = Field(
        description=(
            "True if the image contains a figure perceived through an illusory "
            "or subjective contour; otherwise False."
        )
    )

def get_args_parser():
    parser = argparse.ArgumentParser("Evaluation Script")

    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outs/")
    parser.add_argument("--data_dir", type=str, default="examples/gen100/images")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=8192)

    # Ollama-specific
    parser.add_argument("--max_context", type=int, default=8192)

    # Ollama: enables thinking
    # OpenAI: requests medium reasoning effort
    parser.add_argument("--thinking", action="store_true")

    parser.add_argument("--max_images", type=int, default=1000)

    return parser.parse_args()


def image_to_data_url(image_path):
    """
    Convert a local image to a base64 data URL for the OpenAI API.
    """
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"

def supports_thinking(model):
    info = show(model)
    capabilities = info.capabilities or []

    return "thinking" in capabilities

# --------------------------------------------------
# Ollama
# --------------------------------------------------

def get_ollama_response(
    model,
    query,
    image_path,
    max_tokens=4000,
    thinking=False,
    temperature=0.0,
    max_context=8192,
):
    binary_query = query[1]
    count_query = query[0]
    if thinking:
        thinking = supports_thinking(model)

    response_binary = chat(
        model=model,
        think=thinking,
        messages=[
            {
                "role": "user",
                "content": binary_query,
                "images": [image_path],
            }
        ],
        format=IllusionResponse.model_json_schema(),
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": max_context,
        },
    )

    # Validate Ollama's JSON against the same schema used by OpenAI.
    binary_parsed = IllusionResponse.model_validate_json(
        response_binary.message.content
    )

    response_count = chat(
        model=model,
        think=thinking,
        messages=[
            {
                "role": "user",
                "content": count_query,
                "images": [image_path],
            }
        ],
        format=ImageDescription.model_json_schema(),
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": max_context,
        },
    )

    # Validate Ollama's JSON against the same schema used by OpenAI.
    count_parsed = ImageDescription.model_validate_json(
        response_count.message.content
    )
    combined = {
        **binary_parsed.model_dump(),
        **count_parsed.model_dump(),
    }
    return combined


# --------------------------------------------------
# OpenAI
# --------------------------------------------------

def get_openai_response(
    client,
    model,
    query,
    image_path,
    max_tokens=4000,
    thinking=False,
    temperature=0.0,
):
    request_args = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": query,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                    },
                ],
            }
        ],
        "text_format": ImageDescription,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }

    if thinking:
        request_args["reasoning"] = {
            "effort": "medium"
        }

    response = client.responses.parse(**request_args)

    if response.output_parsed is None:
        raise ValueError(
            f"OpenAI returned no parsed output for {image_path}.\n"
            f"Raw output:\n{response.output_text}"
        )

    return response.output_parsed.model_dump()


# --------------------------------------------------
# Common interface
# --------------------------------------------------

def get_model_response(
    provider,
    model,
    query,
    image_path,
    client=None,
    max_tokens=4000,
    thinking=False,
    temperature=0.0,
    max_context=8192,
):
    if provider == "ollama":
        return get_ollama_response(
            model=model,
            query=query,
            image_path=image_path,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            max_context=max_context,
        )

    elif provider == "openai":
        if client is None:
            raise ValueError("OpenAI client was not initialized.")

        return get_openai_response(
            client=client,
            model=model,
            query=query,
            image_path=image_path,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def run_model(args):

    
    query = ["""
            Inspect the image. Report every visually distinct
            form a human inspecting the image would perceive. Count repeated forms separately.
            Use a short conventional name for each form. Do not explain your answer.
            """,
            "Insepct the image. Is there a true or illusory contour? Do not explain your answer."]

    # Only instantiate the OpenAI client when needed.
    # OPENAI_API_KEY is read automatically from the environment.
    client = OpenAI() if args.provider == "openai" else None

    image_names = [
        name
        for name in sorted(os.listdir(os.path.join(args.data_dir, 'illusory')))
        if os.path.isfile(os.path.join(args.data_dir, 'illusory', name))
    ]

    control_image_names = [
        name
        for name in sorted(os.listdir(os.path.join(args.data_dir, 'control')))
        if os.path.isfile(os.path.join(args.data_dir, 'control', name))
    ]


    if args.max_images is not None:
        image_names = image_names[:args.max_images]

    output_dict = {'illusory':{}, 'control':{}}

    for count, image_name in enumerate(image_names, start=1):

        print(
            f"[{count}/{len(image_names)}] "
            f"{image_name}"
        )

        output_dict['illusory'][image_name] = {}

        full_path = os.path.join(
            args.data_dir,
            'illusory',
            image_name,
        )

        for rep in range(args.reps):

            response = get_model_response(
                provider=args.provider,
                model=args.model,
                query=query,
                image_path=full_path,
                client=client,
                max_tokens=args.max_tokens,
                thinking=args.thinking,
                temperature=args.temperature,
                max_context=args.max_context,
            )

            output_dict['illusory'][image_name][str(rep)] = response

    for count, image_name in enumerate(control_image_names, start=1):

        print(
            f"[{count}/{len(image_names)}] "
            f"{image_name}"
        )

        output_dict['control'][image_name] = {}

        full_path = os.path.join(
            args.data_dir,
            'control',
            image_name,
        )

        for rep in range(args.reps):

            response = get_model_response(
                provider=args.provider,
                model=args.model,
                query=query,
                image_path=full_path,
                client=client,
                max_tokens=args.max_tokens,
                thinking=args.thinking,
                temperature=args.temperature,
                max_context=args.max_context,
            )

            output_dict['control'][image_name][str(rep)] = response

    return output_dict


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    args = get_args_parser()

    output = run_model(args)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    # Avoid problematic characters in filenames.
    safe_model = (
        args.model
        .replace("/", "_")
        .replace(":", "_")
    )

    output_file = os.path.join(
        args.out_dir,
        f"{args.provider}_{safe_model}_response.json",
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
        )

    print(f"Saved results to {output_file}")