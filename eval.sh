for var in "$@"
do
    ./build.sh && ./lsdslam.out "$var" loop && python data/evaluate_rpe.py data/TUM/$var/groundtruth.txt output/"$var"_loop.poses --verbose --fixed_delta --delta_unit f --plot output/"$var"_loop.png > output/"$var"_loop.result
done
