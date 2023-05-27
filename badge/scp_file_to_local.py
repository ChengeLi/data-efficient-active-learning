from glob import glob
import os


folders = ['MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06',
'MNIST_HyperNethypNetBadgePoinKmeans_10_balldim2_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_10_balldim20_c0.06666666666666667',
'MNIST_HyperNetPoincareKmeans_10_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim2_c0.06',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim2',
'MNIST_HyperNet_RiemannianhypNetBadge_50020_dim_ball',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_test',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20',
'MNIST_HyperNetrand_500_balldim20',
'MNIST_HyperNet_hypNetBadge_500Riemannian badge_nofixgradinMLR',
'MNIST_HyperNet_hypNetNorm_500',
'HyperNet_hypNetBadge',
'MNIST_net00_badge_500',
'MNIST_HyperNet_RiemannianhypNetBadge_500100_dim_ball',
'MNIST_HyperNet_hypNetBadge_500',
'MNIST_net0_badge_500',
'MNIST_HyperNet_hypNetBadge_500Riemannian badge']

folders = ['MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_backtoeuclidean', 
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim2_c0.06666666666666667']

folders = ['CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
]

folders = ['MNIST_HyperNethypNetNorm_500_balldim128_c0.06666666666666667']


folders = ['MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667combinedloss',]

folders = ['MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_newnormalized',]

folders = ['MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_normalized']


folders = ['MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667']

# folders = ['CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
# 'CIFAR10_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNethyperEmbPoincareKmeans_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNet3hypNetBadge_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_fc1nofixgrad',
# 'CIFAR10_HyperNet2_dim20hypNetBadge_500',
# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_train_c',
# 'CIFAR10_net00_dim20badge_500',
# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_normalized',
# 'CIFAR10_HyperNethypNetBadge_500_balldim20_c0.06666666666666667_normalized',]

folders = [
'MNIST_HyperNetPoincareKmeans_10_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_10_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_10_balldim2_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim2_c0.06',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim2_c0.06666666666666667',
'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_backtoeuclidean',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_newnormalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr_newlossonly',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr_newlossonly_batchsize250',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667combinedloss',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667combinedloss',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim2',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_test',
'MNIST_HyperNetrand_500_balldim20',
'MNIST_net00_dim20badge_500',

]

folders = ['MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr']

folders = [
'CIFAR10_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',
]


folders = ['MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr']


folders = ['MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr2']

folders = [
 'CIFAR10_HyperNethypNetNorm_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethypNetBadgePoinKmeans_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethyperNorm_plus_Rbadge_100_balldim20_c0.06666666666666667clipr',

## ADD THE CLIPPED VERSIONS
'CIFAR10_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr',
# 'CIFAR10_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr_firstrun',
'CIFAR10_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',


# experiments = [
# 'CIFAR10_net00_dim20badge_500',
'CIFAR10_HyperNethypNetNorm_1000_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethypNetBadgePoinKmeans_1000_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethyperNorm_plus_Rbadge_1000_balldim20_c0.06666666666666667clipr',
 # 'CIFAR10_HyperNetrand_1000_balldim20_c0.06666666666666667clipr',


]

for folder in folders:
	if not os.path.exists('/Users/lichenge/Downloads/badge_output2/{}'.format(folder)):
		os.makedirs('/Users/lichenge/Downloads/badge_output2/{}'.format(folder))
	# os.system('scp -r chenge_p3_new:/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/{}/\*.png {}/'.format(folder, folder))
	os.system('scp -r chenge_p3_new:/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/{}/\*.mp4 {}/'.format(folder, folder))
	# os.system('scp -r chenge_p3_new:/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/{}/\*.txt {}/'.format(folder, folder))
	# os.system('scp -r chenge_p3_new:/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/{}/images/ {}/'.format(folder, folder))
	# os.system('scp -r chenge_p3_new:/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/{}/samples/ {}/'.format(folder, folder))

