# CowBERT: Transformer-Based Cow Identification

## Overview

CowBERT is a BERT-inspired transformer model that learns cow embeddings from temporal proximity sequences. Unlike static embedding methods (Cow2Vec, Graph-BERT), CowBERT captures the **dynamic movement patterns** of cows by modeling their interaction history as sequences.

**Key Insight**: The sequence of cow interactions (trajectory) is the primary signal for identification, not static social preferences or instantaneous network structure.

## Performance Comparison

| Method | Task | Performance |
|--------|------|-------------|
| **CowBERT** | Masked Token Prediction | **~73% Accuracy** |
| Graph-BERT | Masked Node Prediction | ~6% Validation Accuracy |
| Cow2Vec | Context Cow Prediction | ~17% Hit@5 |
| Frequency Baseline | Most Popular Prediction | ~16% Hit@5 |

## Architecture

- **Input**: Temporal sequence of proximity interactions (neighbor lists)
- **Preprocessing**: 
  - Top-K closest neighbors per snapshot (based on RSSI strength)
  - Flattened into linear sequences
  - Chunked to max sequence length
- **Model**: Transformer Encoder with:
  - Positional encoding
  - Multi-head self-attention
  - Feed-forward layers
  - Layer normalization
- **Task**: Masked Language Modeling (MLM)
  - 15% of tokens are masked
  - 80% replaced with [MASK], 10% random token, 10% unchanged
  - Model predicts original cow ID
- **Special Tokens**:
  - `[EGO]`: Identity of the cow whose sequence this is (never masked)
  - `[CLS]`: Sequence start marker
  - `[SEP]`: Sequence end marker
  - `[PAD]`: Padding token
  - `[MASK]`: Masked token

## Latest Setup (December 2024)

### Data Configuration

#### RSSI Data Collection
- **Dataset**: March 17-23, 2025 (7 days)
- **Total Measurements**: 14,017,511 RSSI readings
- **Active Sensors**: 47 sensors deployed
- **Total Cows**: 49 unique animals
  - 47 with RSSI sensor files
  - 2 detected-only (3cf6: 291K detections, 3d04: 289K detections)
- **Signal Frequency**: ~23.18 measurements/second, ~83,438/hour
- **Temporal Patterns**: Higher activity hours 0-8, consistent hours 9-23

#### Graph Generation (Latest Approach)

**Current Method**: 1-minute averaged RSSI windows with NO threshold

```python
def create_1min_network_with_averaging(
    combined_rssi,
    start_time,
    end_time,
    window_seconds=60  # 1-minute windows
    # NO rssi_threshold parameter - include all proximities
):
    """
    Generate network graphs using 1-minute windows with averaged RSSI.
    
    Key Features:
    - 60-second time windows
    - Average RSSI for each cow pair within window
    - NO RSSI threshold filtering (includes all detected proximities)
    - More robust than closest-measurement approach
    - Produces ~9x more edges than 30s closest method
    """
```

**Why This Approach**:
1. **Averaging reduces noise**: Single weak measurements don't create spurious edges
2. **No threshold = complete data**: Includes weak but real proximity signals
3. **1-minute windows**: Balance between temporal resolution and data availability
4. **Robust to gaps**: Averaging handles missing data better than single measurements

**Performance Comparison**:
- **30-second closest measurement** (old): ~60.5 avg edges/graph
- **1-minute averaged, no threshold** (new): ~587.7 avg edges/graph

### Network Sequence Generation

The training pipeline expects temporal graph sequences in pickle format.

#### Generate Network Sequence

```bash
# Generate full network sequence with 1-minute snapshots
python src/generate_full_network_sequence.py \
    --rssi-threshold -100.0 \  # Effectively no threshold (include all)
    --snapshot-time 60 \        # 1-minute windows
    --output-dir network_sequence \
    --start-after-hours 9       # Use hours 9-23 (most consistent data)
```

**Recommended Time Window**: Hours 9-23 for most consistent data (CV < 0.5)

**Output**:
- `network_sequence/network_sequence_rssi-100_YYYYMMDD_HHMMSS.pkl`
- `network_sequence/network_sequence_rssi-100_YYYYMMDD_HHMMSS_metadata.json`
- `network_sequence/network_sequence_rssi-100_YYYYMMDD_HHMMSS_summary.txt`

#### Manually Generate Graphs (Alternative)

If you want custom graph generation with the latest 1-minute averaged approach:

```python
import pandas as pd
import pickle
from pathlib import Path
from create_network_animation import create_1min_network_with_averaging

# Load RSSI data
rssi_dir = Path("RSSI")
all_data = []
for parquet_file in rssi_dir.glob("*_RSSI_*.parquet"):
    df = pd.read_parquet(parquet_file)
    all_data.append(df)
combined_rssi = pd.concat(all_data, ignore_index=True)
combined_rssi['relative_DateTime'] = pd.to_datetime(combined_rssi['relative_DateTime'])

# Select time period (e.g., Day 1, Hours 9-23)
day_1_start = combined_rssi['relative_DateTime'].min()
start_time = day_1_start + pd.Timedelta(hours=9)
end_time = day_1_start + pd.Timedelta(hours=23)

# Generate graphs (1-minute windows, averaged RSSI, no threshold)
temporal_graphs = []
current_time = start_time
while current_time < end_time:
    window_end = current_time + pd.Timedelta(seconds=60)
    G = create_1min_network_with_averaging(
        combined_rssi, 
        current_time, 
        window_end,
        window_seconds=60
    )
    temporal_graphs.append({
        'graph': G,
        'timestamp': current_time,
        'window_start': current_time,
        'window_end': window_end
    })
    current_time = window_end

# Save
with open('network_sequence/my_sequence.pkl', 'wb') as f:
    pickle.dump(temporal_graphs, f)
```

### Excluded Sensors

The following sensors are excluded from training due to data quality issues or sensor failures:

```python
excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
```

These are automatically filtered in the `TopKProximitySequenceExtractor`.

### Training Configuration

**Recommended Hyperparameters**:

```bash
python src/cowbert.py \
    --input network_sequence/network_sequence_rssi-100_*.pkl \
    --output-dir cowbert_out \
    --embedding-dim 128 \
    --epochs 20 \
    --batch-size 64 \
    --top-k 1              # Consider only the closest neighbor per snapshot
```

**Parameter Explanation**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | **Required** | Path to temporal graph sequence pickle file |
| `--output-dir` | `cowbert_embeddings` | Directory for outputs (embeddings, models, metrics) |
| `--embedding-dim` | `128` | Dimensionality of cow embeddings |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `64` | Batch size for training |
| `--top-k` | `1` | Number of closest neighbors to consider per snapshot |

**Why `top-k=1`?**
- Focuses on strongest proximity signals
- Reduces noise from weak/distant detections
- Best results observed with closest neighbor only
- Can experiment with higher values (2, 3, 5)

## Training Pipeline

### Step 1: Activate Environment

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# Or: venv\Scripts\activate  # Windows

# Verify Python version (3.11+ recommended)
python --version
```

### Step 2: Generate Network Sequence

```bash
# Generate 1-minute averaged graphs with no RSSI threshold
# Using hours 9-23 for most consistent data
python src/generate_full_network_sequence.py \
    --rssi-threshold -100.0 \
    --snapshot-time 60 \
    --output-dir network_sequence \
    --start-after-hours 9
```

**Expected Output**:
```
Loading RSSI data from 47 parquet files...
✅ Loaded 14,017,511 measurements
Creating temporal graphs...
✅ Created 840 snapshots
✅ Saved temporal graphs: network_sequence_rssi-100_20241209_143022.pkl
✅ Saved metadata: network_sequence_rssi-100_20241209_143022_metadata.json
```

### Step 3: Train CowBERT

```bash
# Train with recommended settings
python src/cowbert.py \
    --input network_sequence/network_sequence_rssi-100_*.pkl \
    --output-dir cowbert_out \
    --embedding-dim 128 \
    --epochs 20 \
    --batch-size 64 \
    --top-k 1
```

**Training Progress**:
```
======================================================================
COWBERT: TRANSFORMER-BASED COW EMBEDDINGS
======================================================================

Loading temporal graphs...
✅ Loaded 840 snapshots

Extracting proximity sequences (top_k=1)...
Excluding sensors: {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
Processing snapshots: 100%|██████████| 840/840 [00:05<00:00, 168.2it/s]
   Found 43 unique cows (after filtering)

Preparing BERT sequences...
Processing cows: 100%|██████████| 43/43 [00:02<00:00, 18.3it/s]
   Created 12,847 sequences

Training CowBERT...
Epoch 1/20: 100%|██████████| 201/201 [02:15<00:00]
  Train Loss: 3.456, Train Acc: 8.2%, Val Acc: 6.4%
Epoch 2/20: 100%|██████████| 201/201 [02:14<00:00]
  Train Loss: 2.987, Train Acc: 15.3%, Val Acc: 12.8%
...
Epoch 20/20: 100%|██████████| 201/201 [02:13<00:00]
  Train Loss: 0.654, Train Acc: 74.6%, Val Acc: 72.9%

✅ Best Model: Epoch 18, Val Acc: 73.2%
```

### Step 4: Inspect Outputs

```bash
ls cowbert_out/

# Expected files:
# - cow_embeddings.pkl        # Embeddings dictionary
# - cow_embeddings.npy        # NumPy array of embeddings
# - cow_embeddings.csv        # CSV format with cow IDs
# - embedding_metadata.json   # Configuration and cow ID list
# - embeddings_tsne.png       # t-SNE visualization
# - similarity_report.txt     # Nearest neighbor analysis
# - training_metrics.png      # Loss and accuracy curves
# - best_model.pt             # Best checkpoint
```

## Output Files

### Embeddings

**`cow_embeddings.pkl`**: Dictionary mapping cow IDs to embedding vectors
```python
import pickle
with open('cowbert_out/cow_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)  # Dict[str, np.ndarray]
    
# Example: Get embedding for cow '365d'
cow_365d_embedding = embeddings['365d']  # Shape: (128,)
```

**`cow_embeddings.npy`**: NumPy array of all embeddings
```python
import numpy as np
embeddings = np.load('cowbert_out/cow_embeddings.npy')  # Shape: (43, 128)
```

**`cow_embeddings.csv`**: CSV format with cow IDs and embedding dimensions
```
cow_id,dim_0,dim_1,dim_2,...,dim_127
365d,0.234,-0.567,0.891,...,-0.123
365e,-0.345,0.678,-0.234,...,0.456
...
```

### Metadata

**`embedding_metadata.json`**: Configuration and vocabulary
```json
{
  "vocab_size": 43,
  "embedding_dim": 128,
  "cow_ids": ["365d", "365e", "3660", ...],
  "config": {
    "embedding_dim": 128,
    "epochs": 20,
    "batch_size": 64,
    "top_k": 1
  }
}
```

### Visualizations

**`embeddings_tsne.png`**: 2D t-SNE projection of embeddings
- Visualizes embedding space structure
- Colored by cow ID
- Shows clustering patterns

**`training_metrics.png`**: Training curves
- Training loss over epochs
- Training accuracy over epochs
- Validation accuracy over epochs
- Highlights best model checkpoint

**`similarity_report.txt`**: Nearest neighbor analysis
```
COW SIMILARITY REPORT
=====================

Most Similar Pairs (Top 10):
1. 365d ↔ 365e: 0.923
2. 3cf0 ↔ 3cf1: 0.917
3. 3d05 ↔ 3d06: 0.909
...

Individual Cow Neighborhoods:
-----------------------------
Cow: 365d
  1. 365e: 0.923
  2. 3660: 0.876
  3. 3663: 0.834
```

## Model Architecture Details

```python
class CowBERTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
```

**Default Architecture**:
- Embedding dimension: 128
- Attention heads: 8
- Transformer layers: 6
- Feed-forward dimension: 512
- Dropout: 0.1

## Usage Examples

### Load and Use Embeddings

```python
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open('cowbert_out/cow_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Compute similarity between two cows
cow_a = embeddings['365d']
cow_b = embeddings['365e']
similarity = cosine_similarity([cow_a], [cow_b])[0, 0]
print(f"Similarity: {similarity:.3f}")

# Find most similar cows to a target cow
target_cow = '365d'
target_embedding = embeddings[target_cow]

similarities = []
for cow_id, embedding in embeddings.items():
    if cow_id != target_cow:
        sim = cosine_similarity([target_embedding], [embedding])[0, 0]
        similarities.append((cow_id, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print(f"\nMost similar to {target_cow}:")
for cow_id, sim in similarities[:5]:
    print(f"  {cow_id}: {sim:.3f}")
```

### Clustering Analysis

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load embeddings
embeddings_array = np.load('cowbert_out/cow_embeddings.npy')

# Cluster cows
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_array)

# Load cow IDs
import json
with open('cowbert_out/embedding_metadata.json', 'r') as f:
    metadata = json.load(f)
cow_ids = metadata['cow_ids']

# Print clusters
for cluster_id in range(n_clusters):
    cows_in_cluster = [cow_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
    print(f"Cluster {cluster_id}: {', '.join(cows_in_cluster)}")
```

### Downstream Task: Behavior Prediction

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load embeddings
with open('cowbert_out/cow_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Load behavior labels (example: brush visits)
behavior_df = pd.read_csv('brush_visits.csv')  # cow_id, visited_brush

# Prepare data
X = []
y = []
for _, row in behavior_df.iterrows():
    cow_id = row['cow_id']
    if cow_id in embeddings:
        X.append(embeddings[cow_id])
        y.append(row['visited_brush'])

X = np.array(X)
y = np.array(y)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Brush prediction accuracy: {accuracy:.2%}")
```

## Troubleshooting

### Issue: Low Accuracy (<50%)

**Possible Causes**:
1. **Insufficient data**: Check `network_sequence_rssi-*_summary.txt`
   - Need >500 snapshots with activity
   - Avg edges per snapshot should be >100
2. **Poor data quality**: Too many excluded sensors
3. **Wrong time period**: Hours 0-8 have inconsistent data

**Solutions**:
- Use hours 9-23 (`--start-after-hours 9`)
- Lower RSSI threshold to include more proximities (already at -100)
- Increase snapshot duration (`--snapshot-time 120` for 2-minute windows)
- Reduce `--top-k` to 1 for strongest signals only

### Issue: Out of Memory

**Solutions**:
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--embedding-dim` (try 64)
- Reduce temporal graph sequence length (use fewer hours)

### Issue: Training Too Slow

**Solutions**:
- Increase `--batch-size` (if memory allows)
- Reduce number of transformer layers in code (default: 6)
- Use GPU if available (automatically detected by PyTorch)

### Issue: Embeddings Not Separating Well

**Check**:
1. `embeddings_tsne.png` - Should show some clustering
2. `similarity_report.txt` - Top similarities should be >0.7

**Solutions**:
- Increase `--epochs` (try 30-40)
- Increase `--embedding-dim` (try 256)
- Experiment with `--top-k` values (2, 3, 5)
- Ensure using 1-minute averaged graphs (not 30-second closest)

## Data Quality Notes

### Signal Reception Patterns

From comprehensive RSSI analysis (`rssi_signal_frequency_analysis.ipynb`):

- **Total Measurements**: 14,017,511 over 7 days
- **Signal Frequency**: 23.18 measurements/second
- **Temporal Coverage**: 42.7% of intervals have gaps
- **Peak Activity**: Hour 6 (highest signal count)
- **Lowest Activity**: Hour 23 (lowest signal count)

### Hourly Consistency

**Hours 0-8** (Less Consistent):
- Higher signal counts but more variable
- Coefficient of Variation (CV) > 0.5
- More data but less reliable patterns

**Hours 9-23** (Recommended):
- Lower signal counts but consistent
- CV < 0.5 (more predictable)
- Better for training stable embeddings

### Missing Sensors

**Cows without RSSI files** (detected in other sensors' data):
- `3cf6`: 291,374 detections
- `3d04`: 289,092 detections

These cows are **included** in training as they appear in other cows' proximity sequences.

## Advanced Configuration

### Custom Network Generation

If you need different graph generation parameters:

```python
# In src/generate_full_network_sequence.py, modify:

def create_temporal_graphs(
    rssi_data,
    snapshot_time=60,        # Adjust window size
    rssi_threshold=-100.0,   # Adjust threshold (lower = more inclusive)
    use_averaging=True,      # Set to True for averaged RSSI
):
    # Custom logic here
```

### Custom Excluded Sensors

```python
# In src/cowbert.py, modify line ~442:

excluded_sensors = {
    '3668',  # Example: Poor signal quality
    '3cf7',  # Example: Sensor malfunction
    # Add/remove sensors based on your analysis
}
```

### Hyperparameter Tuning

Experiment with different configurations:

```bash
# High-capacity model
python src/cowbert.py \
    --input network_sequence/network_sequence_rssi-100_*.pkl \
    --embedding-dim 256 \
    --epochs 40 \
    --batch-size 32 \
    --top-k 3

# Fast training for testing
python src/cowbert.py \
    --input network_sequence/network_sequence_rssi-100_*.pkl \
    --embedding-dim 64 \
    --epochs 10 \
    --batch-size 128 \
    --top-k 1
```

## References

### Related Documentation

- **Cow2Vec**: `README_COW2VEC.md` - Static embedding baseline
- **Graph-BERT**: `README_GRAPH_BERT.md` - Graph-based approach
- **Contrastive Learning**: `README_CONTRASTIVE.md` - Alternative training approach
- **GNN Models**: `src/README_GNN.md` - Graph Neural Network approaches

### Key Papers

- **BERT**: Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
- **Word2Vec**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations"
- **Transformers**: Vaswani et al. (2017) - "Attention Is All You Need"

## Citation

If you use CowBERT in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information here]

## Contact

For questions or issues:
- GitHub Issues: [Your repository]
- Email: [Your email]

---

**Last Updated**: December 2024  
**CowBERT Version**: 1.0  
**Dataset**: March 17-23, 2025 (7 days, 14M measurements)
