# Agentic AI Farming System Models

This directory contains the trained machine learning models used by the various agents in the Agentic AI farming system. The models are organized by agent type.

## Directory Structure

```
models/
├── crop/               # Crop yield and recommendation models
├── irrigation/         # Irrigation recommendation models
├── market/             # Price prediction models
├── soil/               # Soil analysis models
└── weather/            # Weather prediction models
```

## Model Training

All models are trained using scripts in the `training/` directory. Each agent has its own training scripts that handle:

1. Data preprocessing
2. Model selection and comparison
3. Hyperparameter tuning
4. Model serialization

To train models for all agents:

```bash
python run_agentic_farming.py train
```

To train models for a specific agent:

```bash
python run_agentic_farming.py train --agent irrigation
```

## Model Formats

All models are serialized using `joblib` to maintain compatibility with scikit-learn's API. The models follow a consistent format:

- Main model files: `*_model.joblib` or `*_models.joblib`
- Model metadata: `*_model_metadata.json`
- Performance metrics: `*_model_comparison.json`

## Testing Models

To test the models through their respective agents:

```bash
python run_agentic_farming.py test --agent irrigation
```

To run a full system test:

```bash
python run_agentic_farming.py test --integration
```

## Model Versioning

When updating models, it's recommended to:

1. Backup the current models
2. Date-stamp new model files
3. Update the model metadata with version information

## Sample Models

For development and testing purposes, sample models can be generated:

```bash
python training/generate_sample_models.py
```

These sample models provide minimal functionality for testing the agent interfaces without requiring the full training process.

## Troubleshooting

If models are not loading correctly:

1. Check that the model directory structure matches what the agents expect
2. Verify that the models were trained with compatible versions of scikit-learn
3. Ensure that all required files are present for each model
4. Check the permissions on the model files

## Integration with Gemini LLM

The models provide structured data that is interpreted and enhanced by the Gemini LLM integration. The AgentManager coordinates the interaction between models and the LLM to provide natural language responses to user queries.
