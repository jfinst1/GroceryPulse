import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
import logging
import os
from scipy.interpolate import griddata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
history_file = "training_history.log"
if not os.path.exists(history_file):
    open(history_file, 'a').close()

# Set random seeds and device
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Initializing with device: {device}")

# --- Step 1: Grocery Store Domain Modeling ---
def generate_store_layout(theta, N=64):
    """Generate a detailed 2D grocery store layout with customer traffic perturbations."""
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    a1, a2, a3 = theta  # a1: traffic intensity, a2: traffic frequency, a3: time shift
    
    # Static layout: aisles, coolers, checkout, doors
    layout = np.zeros((N, N))
    
    # Aisles (vertical paths)
    for aisle_x in [10, 20, 30, 40, 50]:
        layout[:, aisle_x-1:aisle_x+2] = 0.2  # Wider aisles
    
    # Coolers (top wall)
    layout[0:6, :] = 0.3
    
    # Checkout counters (bottom with gaps)
    layout[60:64, 0:15] = 0.4  # Counter 1
    layout[60:64, 20:35] = 0.4  # Counter 2
    layout[60:64, 40:55] = 0.4  # Counter 3
    
    # Doors (entrance left, exit right)
    layout[30:34, 0] = 0.5  # Entrance (middle left)
    layout[30:34, -1] = 0.5  # Exit (middle right)
    
    # Dynamic traffic perturbation
    traffic = a1 * np.sin(a2 * np.pi * X + a3) * np.cos(a2 * np.pi * Y)
    x = X + traffic * 0.05
    y = Y
    return x, y, layout + traffic.clip(-0.5, 0.5), layout

def reference_domain(N=64):
    """Generate a reference 2D grid."""
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return np.stack([X, Y], axis=-1)

def diffeomorphism(x, y, theta):
    """Map perturbed store layout back to reference grid."""
    a1, a2, a3 = theta
    X = x - a1 * np.sin(a2 * np.pi * x + a3) * np.cos(a2 * np.pi * y) * 0.05
    Y = y
    return X, Y

def solve_sales_potential(theta, bc, N=64):
    """Solve a sales potential field across the store."""
    x, y, base_layout, _ = generate_store_layout(theta, N)
    u = base_layout.copy()
    u[0, :] = bc[0]  # Top (coolers)
    u[-1, :] = bc[1]  # Bottom (checkout)
    u[:, 0] = bc[2]  # Left (entrance)
    u[:, -1] = bc[3]  # Right (exit)
    
    for _ in range(500):
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
    return u.clip(0, 1)

# --- Step 2: EnhancedMIONet with Quantum Layer ---
class EnhancedMIONet(nn.Module):
    def __init__(self, theta_dim=3, bc_dim=4, hidden_dim=512, num_quantum_weights=6):
        super(EnhancedMIONet, self).__init__()
        self.device = device
        
        self.branch_theta = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.branch_bc = nn.Sequential(
            nn.Linear(bc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.num_quantum_weights = num_quantum_weights
        self.quantum_weights = nn.Parameter(torch.randn(num_quantum_weights, device=self.device) * 0.1)
        self.quantum_dev = qml.device("default.qubit.torch", wires=2, torch_device=device_str)
        
        @qml.qnode(self.quantum_dev, interface='torch')
        def quantum_circuit(inputs, weights):
            inputs = torch.pi * (inputs - inputs.min(dim=1, keepdim=True)[0]) / \
                     (inputs.max(dim=1, keepdim=True)[0] - inputs.min(dim=1, keepdim=True)[0] + 1e-8)
            for i in range(6):
                qml.RY(inputs[..., i], wires=i % 2)
                qml.RX(weights[i], wires=i % 2)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
        
        self.quantum_circuit = quantum_circuit
    
    def quantum_layer(self, inputs):
        z0, z1 = self.quantum_circuit(inputs, self.quantum_weights)
        return ((z0 + z1) / 2).to(dtype=torch.float32)
    
    def forward(self, theta, bc, X_ref):
        batch_size = theta.shape[0]
        n_points = X_ref.shape[-2]
        
        theta_out = self.branch_theta(theta)
        bc_out = self.branch_bc(bc)
        trunk_out = self.trunk(X_ref)
        
        if theta_out.dim() == 3:
            theta_out = theta_out.squeeze(1)
        if bc_out.dim() == 3:
            bc_out = bc_out.squeeze(1)
        
        theta_out = theta_out.unsqueeze(1).expand(-1, n_points, -1)
        bc_out = bc_out.unsqueeze(1).expand(-1, n_points, -1)
        if trunk_out.dim() == 2:
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = theta_out * bc_out * trunk_out
        quantum_input = torch.cat((theta, bc[..., :3]), dim=-1)
        quantum_output = self.quantum_layer(quantum_input)
        final_output = self.final_layer(combined) * (1 + quantum_output.unsqueeze(-1).unsqueeze(-1))
        return final_output.clamp(0, 1)

# --- Step 3: Store Scheduler (QMARL) ---
class StoreScheduler:
    def __init__(self, n_sections=5, n_products=7, n_actions=7, hidden_dim=256):
        self.n_sections = n_sections
        self.n_products = n_products
        self.n_actions = n_actions
        self.device = device
        
        self.qnn_dev = qml.device("default.qubit.torch", wires=self.n_sections, torch_device=device_str)
        self.weights = nn.Parameter(torch.randn(self.n_sections * 3, device=device) * 0.1)
        
        @qml.qnode(self.qnn_dev, interface='torch')
        def qmarl_circuit(states, weights):
            for i in range(self.n_sections):
                qml.RY(states[i, 0], wires=i)
                qml.RX(weights[i * 3], wires=i)
                qml.RZ(weights[i * 3 + 1], wires=i)
                if i < self.n_sections - 1:
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_sections)]
        
        self.qmarl_circuit = qmarl_circuit
        self.optimizer = torch.optim.Adam([self.weights], lr=0.01)
    
    def get_actions(self, states):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        q_values = torch.stack(self.qmarl_circuit(states_tensor, self.weights))
        actions = (q_values * self.n_actions / 2 + self.n_actions / 2).long() % self.n_actions
        return actions.cpu().numpy()
    
    def compute_reward(self, actions, sales_potential, demand_weights):
        reward = 0.0
        for i, action in enumerate(actions):
            if action < self.n_products:
                reward += sales_potential[action] * demand_weights[i]
        return reward / self.n_sections
    
    def train_step(self, states, actions, rewards):
        self.optimizer.zero_grad()
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        q_values = torch.stack(self.qmarl_circuit(states_tensor, self.weights))
        target = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        loss = torch.mean((q_values - target) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def state_dict(self):
        return {
            'weights': self.weights,
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.weights.data = state_dict['weights'].data
        self.optimizer.load_state_dict(state_dict['optimizer'])

# --- Step 4: Integrated Store Optimizer ---
class StoreOptimizer:
    def __init__(self, batch_size=64, n_sections=5, n_products=7):
        self.eq_dimon = EnhancedMIONet().to(device)
        self.scheduler = StoreScheduler(n_sections, n_products)
        self.optimizer = torch.optim.Adam(self.eq_dimon.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.n_sections = n_sections
        self.n_products = n_products
        self.history = []  # Track loss history
    
    def train(self, data, epochs=5, load_from_checkpoint=True):
        checkpoint = "store_optimizer.pth"
        if load_from_checkpoint and os.path.exists(checkpoint):
            self.load_state_dict(torch.load(checkpoint))
            logging.info("Loaded model from checkpoint.")
        
        X_ref_full = reference_domain()
        X_ref_tensor = torch.tensor(X_ref_full.reshape(-1, 2), dtype=torch.float32, device=device)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                theta_batch, bc_batch, _, u_batch = zip(*batch)
                theta_tensor = torch.tensor(np.stack(theta_batch), dtype=torch.float32, device=device)
                bc_tensor = torch.tensor(np.stack(bc_batch), dtype=torch.float32, device=device)
                u_tensor = torch.tensor(np.stack(u_batch), dtype=torch.float32, device=device)
                
                sales_pred = self.eq_dimon(theta_tensor, bc_tensor, X_ref_tensor).squeeze(-1).view(-1, 64, 64)
                
                states = np.random.rand(self.n_sections, 1)
                demand_weights = np.random.rand(self.n_sections)
                actions = self.scheduler.get_actions(states)
                sales_states = sales_pred.mean(dim=(1, 2)).detach().cpu().numpy()[:self.n_products]
                reward = self.scheduler.compute_reward(actions, sales_states, demand_weights)
                
                self.optimizer.zero_grad()
                loss = torch.mean((sales_pred - u_tensor) ** 2)
                loss -= 0.1 * reward
                loss.backward()
                self.optimizer.step()
                
                qmarl_loss = self.scheduler.train_step(states, actions, [reward])
                total_loss += loss.item() + qmarl_loss
            
            avg_loss = total_loss / (len(data) // self.batch_size)
            self.history.append(avg_loss)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            with open(history_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}\n")
        
        torch.save(self.state_dict(), checkpoint)
        logging.info(f"Saved model to {checkpoint}")
    
    def predict(self, theta, bc, X_ref):
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)
        bc_tensor = torch.tensor(bc, dtype=torch.float32, device=device).unsqueeze(0)
        X_ref_tensor = torch.tensor(X_ref.reshape(-1, 2), dtype=torch.float32, device=device)
        with torch.no_grad():
            sales_pred = self.eq_dimon(theta_tensor, bc_tensor, X_ref_tensor).squeeze().cpu().numpy().reshape(64, 64)
        return sales_pred
    
    def state_dict(self):
        return {
            'eq_dimon': self.eq_dimon.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.eq_dimon.load_state_dict(state_dict['eq_dimon'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

# --- Step 5: Data Generation ---
def generate_store_data_worker(args):
    theta, bc = args
    X_ref = reference_domain()
    u = solve_sales_potential(theta, bc)
    x_mapped, y_mapped, _, _ = generate_store_layout(theta)
    u_ref = griddata((x_mapped.flatten(), y_mapped.flatten()), u.flatten(), 
                     (X_ref[..., 0], X_ref[..., 1]), method='cubic', fill_value=0)
    return (theta, bc, X_ref, u_ref)

def generate_store_data(n_samples=300):
    pool = mp.Pool(mp.cpu_count())
    thetas = np.random.uniform([0.1, 0.5, 0], [0.5, 2.0, np.pi], (n_samples, 3))
    bcs = np.random.uniform(0, 1, (n_samples, 4))
    data = pool.map(generate_store_data_worker, zip(thetas, bcs))
    pool.close()
    return data

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time()
    logging.info("Generating store training data...")
    data = generate_store_data(n_samples=300)
    
    logging.info("Training Store Optimizer...")
    store_opt = StoreOptimizer(batch_size=64, n_sections=5, n_products=7)
    store_opt.train(data, epochs=5, load_from_checkpoint=True)
    
    logging.info("Predicting high-sales zones for a test case...")
    theta_test, bc_test, X_ref, u_true = data[0]
    u_pred = store_opt.predict(theta_test, bc_test, X_ref)
    x_test, y_test, _, base_layout = generate_store_layout(theta_test)
    
    # Identify top 5 high-sales locations
    top_indices = np.unravel_index(np.argsort(u_pred.flatten())[-5:], u_pred.shape)
    top_locations = list(zip(top_indices[0], top_indices[1]))
    
    # Visualize store layout, true sales, and predicted sales
    plt.figure(figsize=(18, 5))
    
    # Subplot 1: Store Layout (Floor Map)
    plt.subplot(1, 3, 1)
    plt.contourf(x_test, y_test, base_layout, levels=20, cmap='binary')  # Binary colormap for clarity
    plt.title("Store Floor Map")
    plt.colorbar(label='Structure Intensity')
    plt.text(0.05, 0.5, "Entrance", rotation=90, va='center', ha='center', color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.95, 0.5, "Exit", rotation=90, va='center', ha='center', color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.5, 0.92, "Coolers", va='center', ha='center', color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.5, 0.05, "Checkout", va='center', ha='center', color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    for i, aisle_x in enumerate([11, 21, 31, 41, 51]):
        plt.text(aisle_x / 63, 0.5, f"Aisle {i+1}", va='center', ha='center', color='green', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Subplot 2: True Sales Potential
    plt.subplot(1, 3, 2)
    plt.contourf(x_test, y_test, u_true, levels=20, cmap='viridis')
    plt.title("True Sales Potential")
    plt.colorbar(label='Sales Potential')
    
    # Subplot 3: Predicted Sales Potential with Highlights
    plt.subplot(1, 3, 3)
    plt.contourf(x_test, y_test, u_pred, levels=20, cmap='viridis')
    plt.title("Predicted Sales Potential with Top 5 Locations")
    plt.colorbar(label='Sales Potential')
    for row, col in top_locations:
        plt.plot(col / 63, row / 63, 'ro', markersize=8, label='Top Sales' if row == top_locations[0][0] and col == top_locations[0][1] else "")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot training history
    if store_opt.history:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(store_opt.history) + 1), store_opt.history, 'b-', label='Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Log results
    logging.info(f"Top 5 predicted high-sales locations (row, col): {top_locations}")
    products = ["Milk", "Bread", "Eggs", "Chips", "Soda", "Ice Cream", "Cereal"]
    logging.info("Suggested product placements:")
    for i, (row, col) in enumerate(top_locations):
        if i < len(products):
            logging.info(f"- {products[i]} at ({row}, {col})")
    
    logging.info(f"Total runtime: {time() - start_time:.2f} seconds")
