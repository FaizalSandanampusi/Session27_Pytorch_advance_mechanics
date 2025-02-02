import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom Linear Layer
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * (2 / in_features)**0.5)
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.W + self.b

# Multi-Head Self Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_weights = CustomLinear(embed_dim, embed_dim)
        self.key_weights = CustomLinear(embed_dim, embed_dim)
        self.value_weights = CustomLinear(embed_dim, embed_dim)
        self.output_weights = CustomLinear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        Q = self.query_weights(x)
        K = self.key_weights(x)
        V = self.value_weights(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = attention_weights @ V

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.output_weights(attention_output)

# Custom ReLU
class CustomReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))

# Custom Layer Normalization
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * x_normalized + self.beta

# Custom Feed Forward Network
class CustomFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(CustomFeedForward, self).__init__()
        self.linear1 = CustomLinear(embed_dim, hidden_dim)
        self.activation = CustomReLU()
        self.linear2 = CustomLinear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Custom Mean Pooling
class CustomMeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = CustomLayerNorm(embed_dim)
        self.norm2 = CustomLayerNorm(embed_dim)
        self.feedforward = CustomFeedForward(embed_dim, embed_dim * 4)

    def forward(self, x):
        attention_output = self.self_attention(x)
        x = x + attention_output
        x = self.norm1(x)

        feedforward_output = self.feedforward(x)
        x = x + feedforward_output
        x = self.norm2(x)
        return x

# Custom Transformer Model
class CustomTransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes):
        super(CustomTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.embed = CustomLinear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.pooling = CustomMeanPooling()
        self.fc = CustomLinear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x

def visualize_predictions(model, data_loader):
    model.eval()
    images, labels = next(iter(data_loader))
    images = images.view(images.size(0), 28, 28)

    outputs = model(images)
    _, predictions = torch.max(outputs.data, 1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
        ax.set_title(f'Pred: {predictions[i].item()}\nTrue: {labels[i].item()}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 256
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = CustomTransformerModel(input_dim=28, embed_dim=128, num_heads=8, num_layers=2, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader):
            images = images.view(images.size(0), 28, 28)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), 28, 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Visualize predictions
    visualize_predictions(model, test_loader)

if __name__ == '__main__':
    main()