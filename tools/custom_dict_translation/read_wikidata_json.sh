IN_FILENAME="wikidata_dump.json"
ZIPPED_IN_FILENAME="${IN_FILENAME}.gz"
OUT_FILENAME="custom_dict.json"
URL="http://event.ifi.uni-heidelberg.de/wp-content/uploads/2017/05/WikidataNE_20170320_NECKAR_1_0.json_.gz"
DIR="!!!TODO!!!"

if [ ! -f "${DIR}/${IN_FILENAME}" ]; then
    echo "Downloading ${URL}."
    wget -O $ZIPPED_IN_FILENAME $URL
    echo "Unzipping to ${IN_FILENAME}."
    gunzip $ZIPPED_IN_FILENAME
else
    echo "${IN_FILENAME} already exists."
if

python3 read_wikidata_json.py \
    --input_filepath="${DIR}/${IN_FILENAME}" \
    --output_filepath="${DIR}/${OUT_FILENAME}"
