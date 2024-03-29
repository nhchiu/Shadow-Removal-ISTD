#! /bin/env bash

# download_from_gdrive() {
file_id=$1
file_name=$2
echo $file_id
echo $file_name

# first stage to get the warning html
curl -c /tmp/cookies \
"https://drive.google.com/uc?export=download&id=$file_id" > \
tmp_intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat tmp_intermezzo.html | \
grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies \
"https://drive.google.com$download_link" > $file_name
rm tmp_intermezzo.html
# }
