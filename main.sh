# Write a for loop 
# for i in 10; do
for i in 10 25 50 100 250 500 1000; do
    dir="results_$i"
    mkdir -p $dir
    python3 main.py --epoch $i --dataset WN18RR --save_dir $dir --mode train
    python3 main.py --epoch $i --dataset FB15k-237 --save_dir $dir --mode train
done