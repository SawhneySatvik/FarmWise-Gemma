# Model Training for Agentic AI Farming System

This directory contains scripts for training machine learning models used by the various agents in the Agentic AI farming system. Following the separation of concerns principle, model training is kept separate from agent implementation.

## Benefits of Separating Training from Agents

1. **Cleaner code architecture**: Agents focus solely on inference and domain logic
2. **Reduced production footprint**: Training code isn't deployed unnecessarily
3. **Improved testing**: Agents can be tested independently of training logic
4. **Streamlined deployments**: Models can be updated without changing agent code
5. **Version control of models**: Models can be versioned separately from agent code

## Directory Structure

Each subdirectory corresponds to a specific agent's models:

- `soil/`: Models for soil health analysis and crop recommendation
- `weather/`: Models for rainfall and temperature prediction
- `crop/`: Models for crop yield prediction and growth analysis
- `market/`: Models for price prediction and market analysis
- `irrigation/`: Models for irrigation optimization

## How to Train Models

### Prerequisites

Before training models, make sure you have:

1. Installed all required dependencies from `requirements.txt`
2. Prepared the data by running the preprocessing scripts in the `models` directory
3. Created the necessary directories for storing model outputs

### Training All Models

To train all models at once, run:

```bash
python training/soil/train_models.py
python training/weather/train_models.py
python training/crop/train_models.py
python training/market/train_models.py
python training/irrigation/train_models.py
```

### Training Specific Models

Each training script supports parameters to customize training:

```bash
python training/soil/train_models.py --model soil --output-dir models/soil
```

Available parameters:

- `--model`: Which specific model to train (defaults to 'all')
- `--output-dir`: Directory where trained models will be saved

## Model Outputs

Each training script produces:

1. Serialized model files (`.joblib`)
2. Performance metrics (`.json`)
3. Model metadata for inference

## Updating Agents with New Models

After training new models, the agents will automatically load them at initialization time. No code changes are needed in the agent implementation.

## Adding New Models

To add a new model:

1. Create a new training script in the appropriate subdirectory
2. Implement the training logic following the established pattern
3. Update the agent to load and use the new model

## Troubleshooting

If you encounter issues:

1. Check that data preprocessing completed successfully
2. Verify that all required directories exist
3. Ensure you have sufficient permissions to write model files
4. Check logs for detailed error messages
