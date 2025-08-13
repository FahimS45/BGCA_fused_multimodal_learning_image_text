import os
from torchvision import transforms
from torchvision.models import resnet152
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, DeiTForImageClassification


##================= Load Pre-trained Models =================##


# Load pre-trained ResNet152 model
def load_resnet_model(weight_path):

    resnet_model = resnet152(pretrained=False)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Identity()  
    
    # Load the state_dict
    state_dict = torch.load(weight_path)
    
    # Remove "fc.weight" and "fc.bias" from the state_dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    
    # Load the pruned state_dict into the model
    resnet_model.load_state_dict(state_dict, strict=False)
    resnet_model.eval()  
    
    for param in resnet_model.parameters():
        param.requires_grad = False  
    
    return resnet_model

# Load pre-trained DeiT model
def load_deit_model(weight_path, num_classes=2):

    deit_model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
    
    # Modify the classifier layer
    deit_model.classifier = nn.Identity()  

    # Load trained weights
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    deit_model.load_state_dict(state_dict, strict=False)

    # Set model to evaluation mode
    deit_model.eval()

    # Freeze parameters
    for param in deit_model.parameters():
        param.requires_grad = False  

    return deit_model

# Load the pretrained BERT model
def load_bert_model(weight_path, bert_model_name='dmis-lab/biobert-v1.1'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # Initialize the model with the same configuration used during training
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)  

    # Load the state_dict into the model
    state_dict = torch.load(weight_path, map_location=torch.device('cpu')) 

    # Load the state dict into the model (ignore mismatched keys if any)
    bert_model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    bert_model.eval()

    # Freeze the parameters of the model (for feature extraction)
    for param in bert_model.parameters():
        param.requires_grad = False

    return bert_model, tokenizer


##================= Bidirectional Gated Cross-Attention =================##


class GatedCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, hidden_dim):
        super(GatedCrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(context_dim, hidden_dim)
        self.value_proj = nn.Linear(context_dim, hidden_dim)

        # Gating mechanism
        self.gate_fc = nn.Linear(query_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, context):
        Q = self.query_proj(query).unsqueeze(1)     # [B, 1, H]
        K = self.key_proj(context).unsqueeze(1)     # [B, 1, H]
        V = self.value_proj(context).unsqueeze(1)   # [B, 1, H]

        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, 1, 1]
        attn_weights = self.softmax(attn_scores)       # [B, 1, 1]
        attended = torch.bmm(attn_weights, V).squeeze(1)  # [B, H]

        # Project query into hidden space for fusion
        query_proj = self.query_proj(query)  # [B, H]

        # Gate computation
        gate_input = torch.cat([query, attended], dim=1)  # [B, Q+H]
        gate = self.sigmoid(self.gate_fc(gate_input))     # [B, H]

        # Gated fusion
        gated_output = gate * query_proj + (1 - gate) * attended  # [B, H]
        return gated_output
    

##================= Fully Connected Network =================##


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, output_dim)          
        )

    def forward(self, x):
        return self.fc(x)


##================= Multimodal Model =================##


class MultiModalModelGatedCrossAttention(nn.Module):
    def __init__(self, resnet_model, deit_model, text_model, fc_network):
        super(MultiModalModelGatedCrossAttention, self).__init__()
        self.resnet_model = resnet_model
        self.deit_model = deit_model
        self.text_model = text_model

        self.resnet_dim = 2048
        self.deit_dim = 768
        self.text_dim = 768

        self.fc_network = fc_network

        self.vision_dim = self.resnet_dim + self.deit_dim
        self.hidden_dim = 512  # Fusion hidden space

        # Gated cross attention: both directions
        self.text_to_vision = GatedCrossAttention(self.text_dim, self.vision_dim, self.hidden_dim)
        self.vision_to_text = GatedCrossAttention(self.vision_dim, self.text_dim, self.hidden_dim)

    def forward(self, image_input, input_ids, attention_mask):
        # Extract image features
        resnet_features = self.resnet_model(image_input)                  # [B, 2048]
        deit_features = self.deit_model(image_input).logits              # [B, 768]
        vision_features = torch.cat([resnet_features, deit_features], dim=1)  # [B, 2816]

        # Extract text features from [CLS] token of last hidden state
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        text_features = text_outputs.hidden_states[-1][:, 0, :]          # [B, 768]

        # Gated Cross Attention both directions
        enhanced_text = self.text_to_vision(text_features, vision_features)    # [B, 512]
        enhanced_vision = self.vision_to_text(vision_features, text_features)  # [B, 512]

        # Final fused representation
        fused = torch.cat([enhanced_text, enhanced_vision], dim=1)       # [B, 1024]

        output = self.fc_network(fused)
        return output
    

##================= Create Multimodal Model =================##


# Paths to the models
resnet_weight_path = '/pytorch/default/resnet_best_model.pth'
deit_weight_path = '/pytorch/default/deit_best_model.pth'
bert_weight_path = '/pytorch/default/best_bert_model_state.bin'

# Load 
resnet_model = load_resnet_model(resnet_weight_path)
deit_model = load_deit_model(deit_weight_path)
text_model, bert_tokenizer = load_bert_model(bert_weight_path)

# Fc network
input_dim = 1024  
hidden_dim = 512
output_dim = 2  

fc_network = FullyConnectedNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final model
model = MultiModalModelGatedCrossAttention(
    resnet_model, deit_model, text_model, fc_network
).to(device)