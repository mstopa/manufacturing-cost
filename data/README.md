## Installation
```bash
pip install kaggle
```

## Linux authentication
1. [Generate API token](https://www.kaggle.com/docs/api)

```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## Download the data
```bash
kaggle datasets download -d vinicius150987/manufacturing-cost -p data
unzip data/manufacturing-cost.zip -d data
```