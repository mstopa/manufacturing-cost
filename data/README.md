## Installation
```bash
pip install kaggle
```

## Kaggle authentication
1. [Generate API token](https://www.kaggle.com/docs/api)

```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## Competition authentication
1. Accept [terms and conditions](https://www.kaggle.com/competitions/bosch-production-line-performance)

## Download the data
```bash
kaggle competitions download -c bosch-production-line-performance -p data
unzip data/bosch-production-line-performance.zip -d data
find data -iname '*.zip' | grep -v bosch-production | xargs -I{} unzip {} -d data
```