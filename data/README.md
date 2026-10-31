# Data (not in git)

The OBSCURA dataset (~9.2 GB) may not be redistributed, so it is ignored by git.

## Download

1. Accept the competition rules at https://www.kaggle.com/competitions/obscura
2. Put your Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle > Settings > API > Create New Token).
3. From the repo root:

```bash
bash scripts/download_data.sh
```

Expected layout:

```
data/raw/
  train.csv
  test.csv
  sample_submission.csv
  images/<sample_id>.tif
```

Alternatively set `OBSCURA_DIR=/path/to/folder`. On Kaggle, the code finds `/kaggle/input/obscura` automatically.

See `docs/data_dictionary.md` for column definitions.
