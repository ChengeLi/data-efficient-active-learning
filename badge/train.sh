nquery=500
dataset=CUB #CUB #CIFAR10 #MNIST #
python3 run.py \
--nQuery $nquery \
--data $dataset \
--model vit_small_patch16_224 \
--alg rand \

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
