What Factors Are Associated with Public Attention among Elite Footballers?

MSc Business Analytics dissertation

Data, code and outputs for a study of what predicts online attention among 146 elite male footballers in the 2024/25 season, measured through Instagram followers, Google Trends search interest and Instagram engagement rate.

Folders
01_data_final/           sample frame
02_performance_final/    performance data
03_performance_final/    attention data and final dataset
04_analysis_final/       scripts and outputs
How to run

Requires Python 3 with pandas, numpy, statsmodels, scikit-learn, matplotlib.

Put players_cross_section.csv and google_trends_monthly.csv beside the scripts, then run:

Scripts 
python spec_search.py                  # compares 151 candidate specifications
python analysis.py                     # estimates all models and produces figures
python standardised_coefficients.py    # standardised coefficients

Results are written to analysis_outputs/ (46 files).
