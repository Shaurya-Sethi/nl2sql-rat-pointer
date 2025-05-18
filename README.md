# NL2SQL Model Training

This repository contains code for training a natural language to SQL (NL2SQL) transformer model with comprehensive TensorBoard logging.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nl2sql-model.git
   cd nl2sql-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required data files and model checkpoints.

## Training

### Configuration

The model uses a YAML configuration file for both model architecture and training parameters. TensorBoard logging is configured in the `logging` section of the config:

```yaml
logging:
  tensorboard_log_dir: "runs"  # Base directory for TensorBoard logs
  log_every_n_steps: 10        # Log training metrics every N steps
  log_grad_norm: true          # Whether to log gradient norms
  log_grad_histogram: false    # Whether to log parameter histograms (expensive)
  log_memory: true             # Whether to log memory usage
```

### Running Training

For pretraining:
```bash
python src/train.py --phase pretrain --config src/config.yaml
```

For supervised fine-tuning:
```bash
python src/train.py --phase sft --config src/config.yaml --pretrained_model path/to/pretrained_model.pt
```

## TensorBoard Monitoring

TensorBoard is integrated for comprehensive metric logging during training.

### Logged Metrics

The following metrics are logged:

#### Training Metrics
- Loss (per step and per epoch)
- Perplexity
- Token accuracy
- Learning rate
- Gradient norms
- Memory usage

#### Validation Metrics
- Loss
- Perplexity
- Token accuracy
- Throughput (tokens per second)

#### Model Parameters
- Weight histograms
- Gradient histograms
- Per-layer gradient norms

### Starting TensorBoard Locally

To view training metrics locally:

```bash
# Start TensorBoard server
tensorboard --logdir=runs

# Access TensorBoard in your browser at
# http://localhost:6006
```

You can also compare multiple runs:

```bash
tensorboard --logdir runs_to_compare/run1:runs/20230601-120000_sft,runs_to_compare/run2:runs/20230602-130000_sft
```

### Using TensorBoard on GCP

If your model is training on Google Cloud Platform (GCP), follow these steps to monitor training:

#### Option 1: Port Forwarding
1. SSH into your GCP instance with port forwarding:
   ```bash
   gcloud compute ssh your-instance-name -- -L 6006:localhost:6006
   ```

2. Start TensorBoard on the remote instance:
   ```bash
   tensorboard --logdir=runs --bind_all
   ```

3. Access TensorBoard in your local browser at `http://localhost:6006`

#### Option 2: TensorBoard.dev (For Sharing Results)
1. Install the tensorboard plugin:
   ```bash
   pip install tensorboard-plugin-wit
   ```

2. Upload logs to tensorboard.dev:
   ```bash
   tensorboard dev upload --logdir runs \
     --name "NL2SQL Experiment" \
     --description "Training results for NL2SQL model"
   ```

3. Follow the link provided to view your results online. They'll be available for 90 days.

### Interpreting TensorBoard Metrics

#### Loss and Perplexity
- **Training Loss**: Should steadily decrease
- **Validation Loss**: Should decrease but may plateau
- **Perplexity**: The exponentiated loss value; lower is better

#### Gradient Metrics
- **Gradient Norm**: Measures the magnitude of gradients
  - Very high values (>10) might indicate potential instability
  - Very low values (<0.01) might indicate vanishing gradients
  - Watch for sudden spikes or drops

#### Learning Rate
- Should follow the expected schedule (warmup, decay, etc.)

#### Memory Usage
- Monitor GPU memory to detect leaks or inefficiencies

#### Token Accuracy
- Direction should correlate with loss improvement
- Useful for tracking concrete progress

## Customizing TensorBoard Logging

To modify what's being logged:

1. Edit the TensorBoard configuration in `src/config.yaml`
2. For more detailed changes, modify the logging code in `src/utils/training.py`

## Troubleshooting

- **"No data found"**: Ensure your log directory is correct and that training has saved some data
- **High memory usage**: Set `log_grad_histogram: false` to reduce memory overhead
- **Missing GPU metrics**: Install pynvml with `pip install pynvml`
