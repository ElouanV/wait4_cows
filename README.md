# DATA thèse PACE-COW

Données récoltées dans le cadre de la thèse PACE-COW, en particulier les données des capteurs collier et accéléromètre. Les données sont organisées par type de données. La récolte que je vous partage ici à eu lieu en Mars 2025, du 14 au 27. Je vous présente ici un extrait, du 17 au 23.

Les capteurs collier sont des capteurs de type BLE (Bluetooth Low Energy) qui enregistrent les données de signal RSSI (Radio Signal Strength Indicator) des autres capteurs collier. Les accéléromètres sont des capteurs qui enregistrent les données d'accélération au sein de ce même système.

Les capteurs sont synchronisés entre eux pour que les données soient enregistrées avec le même temps. Le temps utilisé est un temps relatif, c'est-à-dire que le temps est enregistré par rapport à un capteur de référence (le capteur collier) et non pas par rapport à l'heure réelle. Cela permet de réduire la dérive des capteurs et d'avoir des données plus précises.
Colonne à utiliser pour l'utilisation d'un temps relatif aux capteurs est "relative_DateTime".

Tous les capteurs ont un identifiant unique (ID) qui est utilisé pour identifier les capteurs dans les données.

Voici ici la liste des IDs des capteurs utilisés dans cette récolte :
['365d', '365e', '3660', '3662', '3663', '3664', '3665', '3666', '3667', '3668', '3669', '366a', '366b', '366c', '366d', '3ce9', '3cea', '3ceb', '3cec', '3ced', '3cee', '3cef', '3cf0', '3cf1', '3cf2', '3cf3', '3cf4', '3cf5', '3cf7', '3cf8', '3cf9', '3cfa', '3cfb', '3cfc', '3cfd', '3cfe', '3cff', '3d01', '3d02', '3d03', '3d05', '3d06', '3d07', '3d08', '3d09', '3d0c', '3d0f']

Les données sont organisés en fichiers en fonction du type de données :
```
├── data/                          # Raw data directory (gitignored)
    ├── Accelerometer/             # Acceleration data per sensor
    ├── Identification vaches/     # Cow ID correspondence files
    ├── Vidéos/                    # Video recordings of the data collection
    └── RSSI/                      # RSSI signal data per sensor
├── src/                           # Processing, analysis, and visualization code
    ├── generate_temporal_graph_dataset.py  # Generate temporal graph datasets
    ├── gnn_brush_predictor.py     # GNN training for brush prediction
    ├── evaluate_gnn.py            # GNN evaluation and metrics
    ├── utils_paths.py             # Path utilities
    ├── utils_plotting.py          # Plotting utilities
    └── README_GNN.md              # GNN documentation
├── outputs/                       # Generated outputs (gitignored)
    ├── temporal_graphs/           # Temporal graph datasets
    ├── brush_proximity/           # Brush proximity analysis
    └── network_statistics*.csv    # Network statistics
├── brush_proximity_analysis.ipynb # Main analysis notebook
├── temporal_graph_analysis.ipynb  # Temporal graph analysis
├── requirements.txt               # Python dependencies
├── requirements_dev.txt           # Development dependencies
└── README.md
```
# Installation

## Option 1: pip (Virtual Environment)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# Or on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Option 2: conda

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate wait4_cows
```

## Development Setup

### With pip

For development with code formatting and linting tools:

```bash
# Install both production and development dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### With conda

```bash
# Create development environment
conda env create -f environment_dev.yml

# Activate environment
conda activate wait4_cows_dev
```

Development tools included:
- `black`: Code formatter
- `flake8`: Linter
- `pytest`: Testing framework


# Usage

## Generate Temporal Graph Dataset

Create temporal graph snapshots from RSSI data:

```bash
python src/generate_temporal_graph_dataset.py \
  --rssi-threshold -75.0 \
  --snapshot-time 30 \
  --start-after-hours 12 \
  --output-dir outputs/temporal_graphs
```

Parameters:
- `--rssi-threshold`: RSSI threshold in dB (default: -75.0)
- `--snapshot-time`: Snapshot interval in seconds (default: 30)
- `--start-after-hours`: Skip first N hours of data (default: 0)
- `--output-dir`: Output directory (default: outputs/temporal_graphs)

Output files:
- `temporal_graphs_rssi{threshold}_snap{time}s_max_{timestamp}.pkl`: NetworkX graphs
- `temporal_graphs_rssi{threshold}_snap{time}s_max_{timestamp}_metadata.json`: Metadata
- `temporal_graphs_rssi{threshold}_snap{time}s_max_{timestamp}_summary.csv`: Summary statistics

# Formatage des données

Toutes les données sont enregistrées au format parquet pour une meilleure gestion de la mémoire, de la rapidité de lecture des données ainsi que pour une meilleure gestion des types.

Les fichiers auront dans leur nom les informations suivantes (exemple avec `3ce9_accel_elevage_3_cut.parquet`):

- `3d03` : l'ID de l'accéléromètre qui a enregistré les données
- `accel` ou `RSSI` : le type de données
- `elevage_numero` : l'encoding de l'élevage (ici `elevage_3`)
- `cut` : si le fichier est coupé ou non


## Exemple de données

### EXEMPLE DE DATA ACCELERO

|     acc_x |        acc_y |    acc_z |   tick_accel_day |   tick_accel | relative_DateTime          | generated_data   |
|-----------|--------------|----------|------------------|--------------|----------------------------|------------------|
| -0.21875  | -0.000976562 | 0.436523 |           0      |       0      | 2024-07-23 10:26:13        | False            |
| -0.327148 | -0.00195312  | 0.651367 |           0.0625 |       0.0625 | 2024-07-23 10:26:13.062500 | False            |
| -0.380859 | -0.000976562 | 0.761719 |           0.125  |       0.125  | 2024-07-23 10:26:13.125000 | False            |
| -0.410156 |  0           | 0.816406 |           0.1875 |       0.1875 | 2024-07-23 10:26:13.187500 | False            |
| -0.416992 | -0.00292969  | 0.842773 |           0.25   |       0.25   | 2024-07-23 10:26:13.250000 | False            |
| -0.426758 | -0.114258    | 0.918945 |           0.3125 |       0.3125 | 2024-07-23 10:26:13.312500 | False            |

avec :

- `acc_x` l'accélération enregistrée sur l'axe X de l'accéléromètre
- `acc_y` l'accélération enregistrée sur l'axe Y de l'accéléromètre
- `acc_z` l'accélération enregistrée sur l'axe Z de l'accéléromètre
- `tick_accel_day` l'index de l'enregistrement de l'accéléromètre de la journée (un même index apparaitra n fois pour n jours différents)
- `tick_accel` l'index de l'enregistrement de l'accéléromètre GLOBAL (un index apparaitra une seule fois dans le dataset)
- `relative_DateTime` la date et l'heure de l'enregistrement non corrigé pour la drift dûe au capteur
- `generated_data` si la donnée a été générée ou non (30s générée par jour pour combler les trous de données)
- `relative_DateTime` la date et l'heure de l'enregistrement corrigé (synchronistation des capteurs + correction)

### EXEMPLE DE DATA RSSI

|   RSSI |   tick_accel_day |   tick_accel |   ble_id | accelero_id   | relative_DateTime |
|--------|------------------|--------------|----------|---------------|----------------------------|
|    -64 |          12.875  |      12.875  |      106 | 366a          | 2025-07-23 10:10:04.875000 |
|    -61 |          12.9375 |      12.9375 |      254 | 3cfe          | 2025-07-23 10:10:04.937500 |
|    -49 |          13.125  |      13.125  |      238 | 3cee          | 2025-07-23 10:10:05.125000 |
|    -61 |          13.1875 |      13.1875 |      102 | 3666          | 2025-07-23 10:10:05.187500 |
|    -71 |          13.25   |      13.25   |       98 | 3662          | 2025-07-23 10:10:05.250000 |
|    -65 |          13.625  |      13.625  |      251 | 3cfb          | 2025-07-23 10:10:05.625000 |

avec :

- `RSSI` la force du signal enregistrée
- `tick_accel_day` l'index de l'enregistrement de l'accéléromètre de la journée (un même index apparaitra n fois pour n jours différents)
- `tick_accel` l'index de l'enregistrement de l'accéléromètre GLOBAL (un index apparaitra une seule fois dans le dataset)
- `ble_id` l'ID du périphérique BLE qui a été détecté par le capteur enregistrant les données
- `accelero_id` l'ID de l'accéléromètre qui a été détecté par le capteur enregistrant les données
- `relative_DateTime` la date et l'heure de l'enregistrement corrigé

```!Note
le `ble_id` et le `accelero_id` désignent le même capteur, mais le `ble_id` est l'ID du capteur dans le système BLE et l'`accelero_id` est l'ID du capteur dans le système d'accéléromètre
```

```!Note
Dans le cas des données RSSI, on ne génére pas de données pour combler les trous de données mais moins critique que pour les données accéléro ...
```
