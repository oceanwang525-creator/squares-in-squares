#!/bin/bash
D=/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data
mkdir -p $D/zips $D/src
while read s; do
  f=$(echo $s | tr / _)
  [ -f $D/zips/$f.zip.ok ] && continue
  curl -sL --max-time 900 -A curl/8.4.0 -o $D/zips/$f.zip "https://www.kaggle.com/api/v1/datasets/download/$s" -w "$s %{http_code} %{size_download}\n" && touch $D/zips/$f.zip.ok
done < $D/ds_slugs2.txt
echo ALLDONE
