# model_handler.py
import torch
import json
import base64
from PIL import Image
import io
from torchvision import transforms

class ModelHandler:
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def initialize(self, context):
        # Get model file path from context
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model
        serialized_file = context.manifest['model']['serializedFile']
        model_path = f"{model_dir}/{serialized_file}"
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load TorchScript model
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load class names
        with open(f"{model_dir}/index_to_name.json", 'r') as f:
            self.class_names = json.load(f)

    def preprocess(self, data):
        images = []
        for item in data:
            # Handle base64 encoded image
            image_data = item.get("data")
            
            # Decode base64
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)
            
            # Open and process image
            image = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale
            image = self.transform(image)
            images.append(image)
        
        return torch.stack(images).to(self.device)

    def inference(self, data):
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities

    def postprocess(self, inference_output):
        # Get top-k predictions
        top_k = 3
        top_probs, top_indices = torch.topk(inference_output, top_k)
        
        results = []
        for i in range(inference_output.size(0)):
            result = {}
            for j in range(top_k):
                idx = top_indices[i][j].item()
                prob = top_probs[i][j].item()
                result[self.class_names[str(idx)]] = prob
            results.append(result)
        
        return results

    def handle(self, data, context):
        try:
            # Preprocess input
            preprocessed_data = self.preprocess(data)
            
            # Run inference
            inference_output = self.inference(preprocessed_data)
            
            # Postprocess results
            return self.postprocess(inference_output)
        except Exception as e:
            print(f"Error in handle method: {e}")
            raise

# Required for TorchServe
_service = ModelHandler()