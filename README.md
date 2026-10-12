# STDT-Net

Official implementation of **STDT-Net: Lightweight Edge-Based Gesture Sensing Using RGB-D Skeleton Soft Sensor With Spatio-Temporal Decoupled Transformer and Dynamic Graph Filtering**.

## Dependencies

### Requirements

- PyTorch `>=2.4,<3`
- NumPy `>=1.24,<3`
- PyYAML `>=6,<7`

### Installation

```bash
# Clone the repository
git clone https://github.com/QienJane/STDT-Net.git
cd STDT-Net
python -m pip install -r requirements.txt
```

## Data Preparation

### NTU RGB+D 60 / 120

Raw `.skeleton` files and missing-skeleton lists are available from the [official dataset repository](https://github.com/shahroudy/NTURGB-D). `/path/to/...` denotes local paths.

```bash
# NTU RGB+D 60: Cross-Subject (CS)
python -B data/ntu/seq_transformation.py \
  --raw-dir /path/to/ntu60_skeletons \
  --dataset ntu60 --protocol CS \
  --ignored_sample_path /path/to/NTU_RGBD_samples_with_missing_skeletons.txt \
  --output data/processed/ntu60_cs

# NTU RGB+D 60: Cross-View (CV)
python -B data/ntu/seq_transformation.py \
  --raw-dir /path/to/ntu60_skeletons \
  --dataset ntu60 --protocol CV \
  --ignored_sample_path /path/to/NTU_RGBD_samples_with_missing_skeletons.txt \
  --output data/processed/ntu60_cv

# NTU RGB+D 120: Cross-Subject (CS)
python -B data/ntu/seq_transformation.py \
  --raw-dir /path/to/ntu60_skeletons /path/to/ntu120_additional_skeletons \
  --dataset ntu120 --protocol CS \
  --ignored_sample_path /path/to/NTU_RGBD120_samples_with_missing_skeletons.txt \
  --output data/processed/ntu120_cs

# NTU RGB+D 120: Cross-Setup (CE)
python -B data/ntu/seq_transformation.py \
  --raw-dir /path/to/ntu60_skeletons /path/to/ntu120_additional_skeletons \
  --dataset ntu120 --protocol CE \
  --ignored_sample_path /path/to/NTU_RGBD120_samples_with_missing_skeletons.txt \
  --output data/processed/ntu120_ce
```

NTU-120 requires all 120 classes. `--raw-dir` accepts a complete directory or separate NTU-60 and NTU-120 additional-data directories with unique sample IDs. `--ignored_sample_path` points to the local official missing-skeleton list. The converter creates protocol-specific `train/` and `test/` splits.

### RoboGesture-8

RoboGesture-8 contains eight robot commands with 21 hand joints. **The dataset will be released upon acceptance.** Custom recordings follow the format below.

```bash
python -B data/robogesture/prepare.py \
  --manifest /path/to/samples.csv \
  --output data/processed/robogesture8
```

Each source NPZ contains `data` (float32 `[T,21,3]`, raw camera coordinates) and `mask` (boolean or 0/1 `[T,21]`, validity saved during recording).

CSV header:

```csv
sample_id,data_path,label,subject_id,split
```

`data_path` is absolute or relative to the CSV directory. Sample IDs and source paths must be unique; `train`, `val`, and `test` must be nonempty and subject-disjoint. Split assignments are preserved. Labels are zero-based:

| Label | Command | Label | Command |
| --- | --- | --- | --- |
| 0 | Start | 4 | Turn Left |
| 1 | Stop | 5 | Turn Right |
| 2 | Forward | 6 | Accelerate |
| 3 | Backward | 7 | Decelerate |

## Training

| Dataset / protocol | Configuration | Selected checkpoint |
| --- | --- | --- |
| NTU-60 CS | `config/nturgbd-cross-subject/default.yaml` | Final epoch, `last.pt` |
| NTU-60 CV | `config/nturgbd-cross-view/default.yaml` | Final epoch, `last.pt` |
| NTU-120 CS | `config/nturgbd120-cross-subject/default.yaml` | Final epoch, `last.pt` |
| NTU-120 CE | `config/nturgbd120-cross-setup/default.yaml` | Final epoch, `last.pt` |
| RoboGesture-8 | `config/robogesture8/default.yaml` | Best validation accuracy, `best.pt` |

```bash
python -B main.py --config config/nturgbd-cross-subject/default.yaml
python -B main.py --config config/robogesture8/default.yaml
```

Each run requires a separate `--work-dir`; changing `--seed` does not change the output directory.

```bash
python -B main.py --config config/robogesture8/default.yaml \
  --seed 1 --work-dir work_dir/robogesture8/seed1
```

Training saves `config.yaml` and `metrics.jsonl` in `work_dir` and updates `last.pt` each epoch. Checkpoint selection excludes test data.

## Testing

### Evaluation

```bash
python -B main.py --config config/nturgbd-cross-subject/default.yaml \
  --phase eval --checkpoint work_dir/ntu60_cs/seed0/last.pt

python -B main.py --config config/robogesture8/default.yaml \
  --phase eval --checkpoint work_dir/robogesture8/seed0/best.pt
```

### Offline Prediction

Prediction input is one prepared split directory.

```bash
python -B main.py --config config/robogesture8/default.yaml \
  --phase predict --checkpoint work_dir/robogesture8/seed0/best.pt \
  --input /path/to/prepared_clips --output predictions.npz
```

Both modes restore data paths, preprocessing, and quality statistics from the checkpoint. YAML path edits do not redirect inference; `--input` selects a compatible prepared split for prediction.

Both modes save NPZ files containing sample IDs, labels, logits, softmax probabilities, and predictions. `--output` sets the destination; the default is the checkpoint directory.

## Project Structure

```text
STDT-Net/
├── config/             # Dataset and training configurations
├── data/               # Data preparation scripts and generated arrays
├── ensemble/           # Optional score fusion
├── feeders/            # Skeleton loading and preprocessing
├── graph/              # Skeleton topologies
├── model/              # STDT-Net model implementation
├── work_dir/           # Generated training and evaluation outputs
├── main.py             # Training, evaluation, and prediction entry point
├── requirements.txt
└── README.md
```

## Acknowledgements

STDT-Net builds on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). We thank the authors for their work.

## Contact

For any questions, please contact zhanqq@mail2.sysu.edu.cn
