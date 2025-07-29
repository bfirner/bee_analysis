#!/usr/bin/bash

# Generate a test video.
# Generate images with imagemagick's convert command and join them into a video.
# Each image will have the frame number, a color pattern, and markings to indicate different points
# in the image that are useful for testing the data preparation pipeline.

# Videos are 1280x720.
size="1280x720"

if [[ $# -ge 2 ]]; then
    outpath="$2"
else
    outpath="."
fi

echo "video_timestamp_relative_ns, odd" > "$outpath/labels.csv"
for num in $(seq $1); do
    convert -background white -fill black -gravity center -size $size -pointsize 24 label:$num "$outpath/synthetic_test_$num.png"
    # Label 0 for even and 1 for odd
    oddness=$(($num % 2))
    echo "$((num * 33000)), $oddness" >> "$outpath/labels.csv"
done
ffmpeg -r  30 -s $size -i "$outpath/synthetic_test_%d.png" -vcodec libx264 -crf 20 "$outpath/synthetic_test_video.mp4"
# Leave the images around for testing
#rm synthetic_test_*.png
