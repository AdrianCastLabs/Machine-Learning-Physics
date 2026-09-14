# NBody-Neural-Network

A simple machine learning project that trains neural networks to predict physics-based motion, including a two-body gravity simulation and a bouncing ball simulation.

## Overview

This project explores how neural networks can learn motion from generated simulation data. It includes scripts for:

- Generating physics simulation datasets
- Training PyTorch models on those datasets
- Saving model predictions to CSV files
- Visualizing predicted motion with Pygame

The main focus is an **N-body gravity prediction model**, where a neural network learns to estimate how two celestial bodies would behave in space.

## Features

- Two-body gravity simulation data generation
- Neural network training with PyTorch
- Multi-frame prediction 
- CSV data and prediction storage
- Pygame visualizers for viewing predicted motion
- Simple bouncing ball example for experimenting with learned physics

## Usage (inside the project's root directory)

Install the required Python packages:
``` bash
pip install -r requirements.txt
```

1. Generate gravity simulation data
``` bash
python simulations/GravitySimulation.py
```

This creates a CSV dataset in the data/ folder.
2. Train the gravity model
``` bash
python training/GravityTraining.py
```

This trains a neural network on the generated simulation data and saves a checkpoint in the models/ folder.
3. Generate predictions
``` bash
python generate/PredictGravity.py
```

This loads the trained model and writes predicted motion data to the predictions/ folder 

(as long as the model is trained you can run this script as much as you want to generate different results).
4. Visualize the gravity prediction
``` bash
python visuals/GravitySimulationVisualizer.py
```

A Pygame window will open and display the predicted movement of the two bodies.

