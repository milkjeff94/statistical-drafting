# Model Refresh Automation

This directory contains scripts that automate the process of checking for and downloading new Magic: The Gathering draft data from 17lands, then retraining models when updates are detected.

## Overview

The automation system performs the following tasks:

1. **Checks for new MTG sets** - Uses web scraping to detect when new sets are released
2. **Downloads updated card data** - Fetches the latest cards.csv when new sets are available
3. **Monitors draft data updates** - Tracks when Premier and Traditional draft datasets are updated
4. **Downloads new draft data** - Fetches updated draft data files to the local data directory
5. **Triggers model retraining** - Automatically runs the training pipeline when new data is available
6. **Tracks update history** - Maintains a JSON log of when data was last updated

## Rate Limiting & Efficiency

Efforts have been made to minimize the number of HTTP requests:

- **HEAD requests** are used to check file modification dates without downloading content
- **Conditional downloads** only occur when data has actually been updated
- **JSON tracking** prevents redundant checks and downloads
- **Batch operations** group related tasks together

## Files

- `README.md` - This documentation file
- `data_tracker.json` - JSON file tracking the most recent set and update dates
- `refresh_models.py` - Main automation script
- `requirements.txt` - Python dependencies for the automation scripts

## Usage

Run the automation script:

```bash
python refresh_models.py
```

The script will:
1. Check if new sets or data updates are available
2. Download any new data that's detected
3. Retrain models for updated datasets
4. Update the tracking JSON with new timestamps

## Data Sources

- **17lands Public Datasets**: https://www.17lands.com/public_datasets
- **Cards Data**: https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv
- **Draft Data**: S3 URLs discovered dynamically from the 17lands website

## Dependencies

The automation requires the statistical-drafting package and its dependencies, plus:
- `requests` for HTTP operations
- `playwright` for web scraping (already in get_latest_set.py)

## Automation Schedule

This script is designed to be run periodically (e.g., daily or weekly) via cron or similar scheduling systems to keep models up-to-date with the latest draft data.
