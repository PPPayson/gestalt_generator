import os 
import json
import argparse
from ollama import chat
from pydantic import BaseModel

MODELS = ["llama3.2-vision:11b", "llama3.2-vision:90b"]

class Form(BaseModel):
    name: str
    confidence: float

class ImageDescription(BaseModel):
    illusion: bool
    count: int
    forms: list[Form]


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation Script', add_help=False)
    print(parser)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outs/')
    parser.add_argument('--data_dir', type=str, default='data/silhouettes')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--max_context', type=int, default=8192)
    parser.add_argument('--thinking', default=False, action='store_true'),
    return parser.parse_args()

def get_model_response(model, query, image_path, max_tokens=4000, thinking=True, temperature=0.0, max_context=8192):
    response = chat(
        model = model,
        'thinking': thinking,
        messages = [
            {
            'role': 'user',
            'content': query,
            'images': [image_path]
            }],
        format = ImageDescription.model_json_schema(),
        options={'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': max_context}
        )
    return response

def run_model(args):
    model = args.model
    output_dict = {}
    if model not in MODELS:
        print(f"Unsupported model {model}")
        return
    query = """
            Inspect the image. Is there an illusory contour? Report every visually distinct form a human inspecting the image would perceive.
            Count repeated forms separately. Use a short conventional name for each form. Do not explain your answer. 
            Respond only in JSON:
            """
    max_images = 100
    for count, image_path in enumerate(os.listdir(args.data_dir)):
        output_dict[image_path] = {}
        full_path = os.path.join(args.data_dir, image_path)
        for i in range(args.reps):
            response = get_model_response(model,
                            query, full_path,
                            max_tokens=args.max_tokens,
                            thinking=args.thinking,
                            temperature=args.temperature,
                            max_context=args.max_context)
            output_dict[image_path][i] = response
        if count >= max_images:
            break
    
    return output_dict

if __name__ == "__main__":
    args = get_args_parser()
    model = args.model
    output = run_model(args)
    os.makedirs(args.out_dir, exist_ok=True)
    output_file = os.path.join(args.out_dir, f"{model}_response.json")
    with open(args.out_dir, 'w') as f:
        json_content = json.dumps(output, indent=4)
        f.write(json_content)

