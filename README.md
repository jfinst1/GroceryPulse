# GroceryPulse - An eQ-DIMON Grocery Store Simulator

This project adapts a hybrid quantum-classical deep learning framework to simulate a grocery store and predict high-selling item locations in a 2D space. Originally inspired by a Space-Air-Ground Integrated Network (SAGIN) optimizer, the simulator leverages PDE-based sales potential modeling, a quantum-enhanced neural network (`EnhancedMIONet`), and a quantum multi-agent reinforcement learning (QMARL) scheduler to optimize product placement.

## Overview

The simulator models a grocery store as a 64x64 grid containing aisles, coolers, and checkout zones. It predicts sales potential across the store based on customer traffic patterns and boundary conditions, then suggests optimal locations for high-demand products such as milk, bread, and eggs.

### Key Features
- **Store Layout**: 64x64 grid with static store features (aisles, coolers, checkout zones) and dynamic customer traffic perturbations.
- **Sales Prediction**: `EnhancedMIONet`, a neural network incorporating a 2-qubit quantum layer, predicts sales potential.
- **Product Placement**: Quantum multi-agent reinforcement learning (`StoreScheduler`) assigns products to optimal sales locations.
- **Visualization**: Generates contour plots comparing true vs. predicted sales potential.

## Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**:
  - `numpy`
  - `torch`
  - `pennylane`
  - `matplotlib`
  - `scipy`

Install dependencies using pip:
```bash
pip install numpy torch pennylane matplotlib scipy
```

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/eq-dimon-grocery.git
cd eq-dimon-grocery
```

Ensure dependencies are installed (see above).

Save the main script as `grocerypulse_v1.py` (or use the provided file if already included).

## Usage

Run the simulator:
```bash
python grocerypulse_v1.py
```

## What to Expect
- **Training**: The model trains on 300 synthetic store scenarios over 5 epochs, logging progress.
- **Output**:
  - Contour plots displaying true vs. predicted sales potential.
  - A list of the top 5 high-sales locations with suggested product placements.

**Example log output:**
```
2025-04-07 10:00:00 - INFO - Top 5 predicted high-sales locations (row, col): [(63, 32), (62, 30), (1, 40), (0, 20), (63, 50)]
2025-04-07 10:00:00 - INFO - Suggested product placements:
2025-04-07 10:00:00 - INFO - - Milk at (63, 32)
2025-04-07 10:00:00 - INFO - - Bread at (62, 30)
...
```

## Customization
- **Store Layout**: Modify `generate_store_layout` in the script to adjust aisles, coolers, or add new features.
- **Products**: Change the product list in the main execution block to target specific items.
- **Traffic Patterns**: Adjust theta ranges in `generate_store_data` for various scenarios (e.g., increased weekend traffic).

## How It Works
- **Data Generation**: Synthetic store data created with varying traffic patterns (`theta`) and boundary sales conditions (`bc`).
- **Sales Potential**: PDE-based diffusion solver generates a true sales field for model training.
- **Prediction**: `EnhancedMIONet` uses classical neural networks and quantum layers to predict sales potential.
- **Scheduling**: `StoreScheduler` employs quantum reinforcement learning (QMARL) to optimally assign product locations.
- **Output**: Visualization of top sales locations through contour plots.

## Limitations
- **Synthetic Data**: Currently uses randomized synthetic data; real-world data would enhance accuracy.
- **Quantum Scale**: Limited quantum components (2 qubits prediction, 5 qubits scheduling) are simulated classically; real quantum hardware could further improve performance.
- **Simplifications**: Basic diffusion-based PDE model; future versions could incorporate complex dynamics like explicit customer paths.

## Future Enhancements
- Integration of real grocery store sales data.
- Expansion of quantum circuits for increased complexity.
- Interactive features for user-defined store layouts.
- Additional metrics such as prediction accuracy and sales uplift.

## Credits

Inspired by my previous works and pushing the limits of python. Special thanks to the open-source communities behind PyTorch, PennyLane, and SciPy.

## License

This project is licensed under my craziness. See the `LICENSE` file for details.