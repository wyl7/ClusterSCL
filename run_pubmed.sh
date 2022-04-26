data_name='pubmed'
data_lr=0.01
data_device=1
#========================================CE========================================
for num_per_class in 20 40 60
do
    for learning_rate in 0.1 0.01 0.001
    do 
        python train_ce.py --dataset $data_name --learning_rate $learning_rate --seed 0 --train_nodes_per_class $num_per_class
    done
done

#========================================SupCon========================================
for num_per_class in 20 40 60
do
    for pretrain_epochs in $(seq 1 1 30)
    do
        python train_supcon.py --device $data_device --learning_rate $data_lr --dataset $data_name --batch_size 32 --pretrain_epochs $pretrain_epochs --seed 0 --train_nodes_per_class $num_per_class
    done
done

#========================================ClusterSCL========================================
for num_per_class in 20 40 60
do
    for num_cluster in $(seq 3 1 6)
    do
        for pretrain_epochs in $(seq 1 1 30)
        do
            for kappa in $(seq 0.05 0.02 0.09)
            do
                for eta in $(seq 0.05 0.05 0.1)
                do
                    python train_clusterscl.py --device $data_device --dataset $data_name --learning_rate $data_lr --batch_size 32 --pretrain_epochs $pretrain_epochs --num_cluster $num_cluster --kappa $kappa --eta $eta --seed 0 --train_nodes_per_class $num_per_class
                done
            done
        done
    done
done

