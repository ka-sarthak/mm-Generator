#!/bin/sh

# for experiment in /Users/kapoor/Articles/article1_FNO_UNet/data/elasto_plastic/testing/256*grains/;
# do 
# echo "running inference for $(basename -- $experiment)"
# python inference_test_cases.py "$(basename -- $experiment)"
# done

for experiment in 256_basicFFT #64 128 256 512 1024 2048
do
echo "running inference for $experiment"
python inference_test_cases.py "$experiment"
done


# 256_oneppt_circular 256_oneppt_diamond 256_oneppt_square
# 64 128 256 512 1024 2048
# 256_*grains
# 2048_1280grains
# aspect ratio