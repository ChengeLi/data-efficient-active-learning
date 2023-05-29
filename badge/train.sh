nquery=300
nstart=6000
dataset=CIFAR100 #CUB #MNIST #
python3 run.py \
--data $dataset \
--nQuery $nquery \
--nStart $nstart \
--model vit_small_patch16_224 \
--aug 1 \
--alg badge \

# --alg hypNetBadgePoinKmeans

# --alg rand \
# --alg hyperNorm_plus_Rbadge \
# --alg hypNetNorm \


# --alg hyperNorm_plus_Rbadge \





# --model resnet50 \
# --alg badge
# --alg hypNetBadgePoinKmeans



# --alg hyperEmbPoincareKmeans \






# --model net00 \
# --alg badge

# --alg hypNetBadgePoinKmeans 

# --alg PoincareKmeans



# --alg hypNetBadge \

# --alg hyperNorm_plus_Rbadge \




# --alg hypNetNorm \



# 
# 
# --alg hypNetBadge \

# --model net0
# --model net00

# --alg badge \


#--model mlp
#--model resnet
#--model HyperNet

#runs an active learning experiment using a ResNet and CIFAR-10 data, querying batches of 1,000 samples according to the BADGE algorithm.
#This code allows you to also run each of the baseline algorithms used in our paper.

