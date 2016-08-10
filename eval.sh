for var in "$@"
do
    make -j4 && ./lsdslam.out "$var" loop && python evaluate_rpe.py data/TUM/$var/groundtruth.txt output/"$var"_loop.poses --verbose --fixed_delta --delta_unit f --plot output/"$var"_loop.png > output/"$var"_loop.result
done
