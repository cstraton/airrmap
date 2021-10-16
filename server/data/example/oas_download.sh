# Populate with wget then add permissions: $ chmod u+rx oas_download.sh
# -nc: Won't download again if file already exists (in case of failing halfway through)
# Will save to ./src/seq/

# Examples (single data unit):
# JSON:  wget http://opig.stats.ox.ac.uk/webapps/ngsdb/json/Bashford_2013/Bashford_2013_CLL_subject-1_Bulk_CLL_subject-1_Age-77_repeat-1_iglblastn_Bulk.json.gz -nc -P ./src/seq/
# CSV:   wget http://opig.stats.ox.ac.uk/webapps/ngsdb/json/Roshkin_2020/700010094_igblastn_anarci_Heavy_Bulk.csv.gz -nc -P ./src/seq/