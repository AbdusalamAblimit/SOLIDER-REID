#!/bin/bash
# Download VOC2012 dataset for Realistic Occlusion Augmentation (ROA)
# Usage: bash scripts/download_voc.sh [data_dir]

DATA_DIR="${1:-data}"
VOC_DIR="$DATA_DIR/VOCdevkit"

if [ -d "$VOC_DIR/VOC2012/JPEGImages" ]; then
    echo "VOC2012 already exists at $VOC_DIR/VOC2012"
    echo "Found $(ls $VOC_DIR/VOC2012/JPEGImages/*.jpg 2>/dev/null | wc -l) images"
    exit 0
fi

echo "Downloading VOC2012..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Try multiple mirrors
URLS=(
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
)

DOWNLOADED=0
for URL in "${URLS[@]}"; do
    echo "Trying: $URL"
    if wget -q --show-progress --timeout=60 "$URL" -O VOCtrainval_11-May-2012.tar; then
        DOWNLOADED=1
        break
    else
        echo "Failed, trying next mirror..."
        rm -f VOCtrainval_11-May-2012.tar
    fi
done

if [ $DOWNLOADED -eq 0 ]; then
    echo "ERROR: Failed to download VOC2012 from all mirrors."
    echo "Please manually download VOCtrainval_11-May-2012.tar and extract to $DATA_DIR/"
    exit 1
fi

echo "Extracting..."
tar xf VOCtrainval_11-May-2012.tar
rm -f VOCtrainval_11-May-2012.tar

if [ -d "$VOC_DIR/VOC2012/JPEGImages" ]; then
    echo "VOC2012 ready at $VOC_DIR/VOC2012"
    echo "Found $(ls $VOC_DIR/VOC2012/JPEGImages/*.jpg | wc -l) images"
else
    echo "ERROR: Extraction failed"
    exit 1
fi
