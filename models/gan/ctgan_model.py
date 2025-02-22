# models/gan/ctgan_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super(Generator, self).__init__()
        
        # Build the network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # No activation for the output layer as we'll use tanh for continuous
        # and gumbel softmax for categorical separately
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256]):
        super(Discriminator, self).__init__()
        
        # Build the network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        # No sigmoid here - we'll use BCEWithLogitsLoss which includes sigmoid
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class CTGAN:
    def __init__(self, embedding_dim=128, generator_dims=[256, 256], 
                 discriminator_dims=[256, 256], batch_size=500,
                 epochs=300, learning_rate=2e-4):
        self.embedding_dim = embedding_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # These will be set when fitting the model
        self.generator = None
        self.discriminator = None
        self.preprocessor = None
        self.data_dim = None
        
    def fit(self, data, preprocessor=None):
        """
        Fit the CTGAN model to the data
        
        Args:
            data (DataFrame): The data to fit
            preprocessor: A data preprocessor that transforms the data
        """
        if preprocessor:
            self.preprocessor = preprocessor
            transformed_data = self.preprocessor.transform(data)
        else:
            transformed_data = data
            
        self.data_dim = transformed_data.shape[1]
        
        # Initialize generator and discriminator
        self.generator = Generator(
            self.embedding_dim, 
            self.data_dim,
            self.generator_dims
        ).to(self.device)
        
        self.discriminator = Discriminator(
            self.data_dim,
            self.discriminator_dims
        ).to(self.device)
        
        # Set up optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor(transformed_data.values).to(self.device)
        dataset = torch.utils.data.TensorDataset(data_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.epochs):
            d_losses = []
            g_losses = []
            
            for real_data in dataloader:
                real_data = real_data[0]  # Unpack the batch
                batch_size = real_data.size(0)
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                # Real data
                d_real = self.discriminator(real_data)
                d_real_loss = torch.nn.BCEWithLogitsLoss()(d_real, torch.ones_like(d_real))
                
                # Fake data
                z = torch.randn(batch_size, self.embedding_dim).to(self.device)
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data.detach())
                d_fake_loss = torch.nn.BCEWithLogitsLoss()(d_fake, torch.zeros_like(d_fake))
                
                # Combined loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                d_losses.append(d_loss.item())
                
                # Train generator
                g_optimizer.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, self.embedding_dim).to(self.device)
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data)
                
                # Generator wants discriminator to think data is real
                g_loss = torch.nn.BCEWithLogitsLoss()(d_fake, torch.ones_like(d_fake))
                g_loss.backward()
                g_optimizer.step()
                g_losses.append(g_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D Loss = {np.mean(d_losses):.4f}, G Loss = {np.mean(g_losses):.4f}")
        
        return self
    
    def generate(self, n_samples):
        """Generate synthetic samples"""
        self.generator.eval()
        
        # Generate samples in batches to avoid memory issues
        synthetic_data = []
        remaining = n_samples
        
        with torch.no_grad():
            while remaining > 0:
                current_batch = min(self.batch_size, remaining)
                z = torch.randn(current_batch, self.embedding_dim).to(self.device)
                fake_data = self.generator(z).cpu().numpy()
                synthetic_data.append(fake_data)
                remaining -= current_batch
        
        # Combine batches
        synthetic_array = np.vstack(synthetic_data)
        
        # Convert to DataFrame if preprocessor exists
        if self.preprocessor:
            synthetic_df = pd.DataFrame(
                synthetic_array,
                columns=self.preprocessor.transform(pd.DataFrame()).columns
            )
            # TODO: Implement inverse_transform properly
            return synthetic_df
        else:
            return synthetic_array