# Populate with wget then add permissions: $ chmod u+rx oas_download.sh
# -nc: Won't download again if file already exists (in case of failing halfway through)
# Will save to ./src/seq/

# Examples (single data unit):
# CSV:   wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz -nc -P ./src/seq/
