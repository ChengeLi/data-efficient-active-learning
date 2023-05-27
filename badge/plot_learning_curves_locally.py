import matplotlib.pyplot as plt
from glob import glob
import os

# root = '/Users/lichenge/Downloads/badge_output/'
root = '/Users/lichenge/Downloads/badge_output2/'
folderlist = glob(os.path.join(root, './*'))
experiments = [folder.split('/')[-1] for folder in folderlist]


# experiments = [
# 'MNIST_HyperNet_RiemannianhypNetBadge_50020_dim_ball',
#  'MNIST_HyperNetrand_500_balldim20',
#  # 'MNIST_HyperNet_RiemannianhypNetBadge_500100_dim_ball',

#  # 'MNIST_net0_badge_500',
#  # 'MNIST_net00_badge_500',

#  # 'HyperNet_hypNetBadge',
#  # 'MNIST_HyperNet_hypNetBadge_500',
#  # # 'MNIST_HyperNet_hypNetBadge_500Riemannian badge',#double fix grad
#  # 'MNIST_HyperNet_hypNetBadge_500Riemannian badge_nofixgradinMLR',
#  # 'MNIST_HyperNet_hypNetNorm_500',
#  # 'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim2',

#  ]




experiments = [
'CIFAR10_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized',


# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_normalized',
# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
'CIFAR10_HyperNethyperEmbPoincareKmeans_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_train_c',

# 'CIFAR10_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_newnormalized_fc1nofixgrad',
# 'CIFAR10_HyperNet2_dim20hypNetBadge_500',
'CIFAR10_HyperNet3hypNetBadge_500_balldim2_c0.06666666666666667_newnormalized',

# 'CIFAR10_net00_dim20badge_500',
# 'CIFAR10_HyperNethypNetBadge_500_balldim20_c0.06666666666666667_normalized',

## ADD THE CLIPPED VERSIONS
'CIFAR10_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr_firstrun',
'CIFAR10_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethypNetBadgePoinKmeans_1000_balldim20_c0.06666666666666667clipr',
'CIFAR10_HyperNethypNetNorm_1000_balldim20_c0.06666666666666667clipr',

 'CIFAR10_HyperNethyperNorm_plus_Rbadge_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethypNetNorm_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethyperNorm_plus_Rbadge_1000_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNetrand_1000_balldim20_c0.06666666666666667clipr',

] 
# plt.legend([ 'HyperNet + HyperNorm (Poincare ball dimension = 2)', 
#              'HyperNet + HyperBADGE (Poincare ball dimension = 2)',
#              'HyperNet(extra linear layer) + HyperBADGE (Poincare ball dimension = 2)',
             
#              'HyperNet-Clipped + HyperNorm (Poincare ball dimension = 20)',  
#               'HyperNet-Clipped + HyperBADGE (Poincare ball dimension = 20)',
#               'HyperNet-Clipped + half half (Poincare ball dimension = 20)', 
#               ])
plot_results(experiments, datasetname='CIFAR10')



experiments = [
'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667',

# 'MNIST_HyperNetPoincareKmeans_10_balldim20_c0.06666666666666667',
# 'MNIST_HyperNethypNetBadgePoinKmeans_10_balldim20_c0.06666666666666667',
# 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667',
# 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667_normalized',
# 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim2_c0.06',
# 'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667',
# 'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667_normalized',
# 'MNIST_HyperNethypNetBadge_500_balldim2_c0.06',
'MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_backtoeuclidean',
# 'MNIST_HyperNethypNetBadge_500_balldim2_c0.06666666666666667_normalized',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_newnormalized',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_normalized',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr_newlossonly',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr_newlossonly_batchsize250',
# 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667combinedloss',
# 'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized',
# 'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized_oldnormtovis',
# 'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667combinedloss',
# 'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim2',## although good in the middle, the net needs to be clipped
# 'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20',
# 'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_test',
# 'MNIST_HyperNetrand_500_balldim20',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667',

]
# plt.legend([ 'HyperNet + HyperNorm (Poincare ball dimension = 2)', 'HyperNet + HyperNorm (Poincare ball dimension = 20)',  
#             'HyperNet + HyperBADGE (Poincare ball dimension = 2)', 'HyperNet + HyperBADGE (Poincare ball dimension = 20)'])




## ball 20
experiments = [
'MNIST_net00_dim20badge_500',
'MNIST_HyperNetrand_500_balldim20',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667',
## not used 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_newnormalized',
## not used 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_normalized',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
## not used 'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',

] # plt.legend([ 'Net + BADGE', 'HyperNet + Random', 'HyperNet + HyperNorm', 'HyperNet + HyperBADGE', 'HyperNet-Clipped + HyperBADGE', 'HyperNet-Clipped + HyperBADGE + HyperNorm'])
plot_results(experiments, datasetname='MNIST')






## new for supp

experiments = [
'MNIST_net00_dim20badge_500',
'MNIST_HyperNetrand_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',

# 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr2', #didn't finish
]
plot_results(experiments, datasetname='MNIST, batch size=500')
plt.legend([ 'Net + BADGE',
         'HyperNet-Clipped + Random',
         'HyperNet-Clipped + HyperNorm',
         'HyperNet-Clipped + HyperBADGE',
         'HyperNet-Clipped + HyperBADGE + HyperNorm'])




experiments = [
'MNIST_HyperNetrand_100_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetNorm_100_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetBadgePoinKmeans_100_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethyperNorm_plus_Rbadge_100_balldim20_c0.06666666666666667clipr',

'MNIST_HyperNetrand_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr',
'MNIST_HyperNethyperNorm_plus_Rbadge_500_balldim20_c0.06666666666666667clipr',


]
plot_results(experiments, datasetname='MNIST')
plt.legend([ 
         'HyperNet-Clipped + Random (batch size=100)',
         'HyperNet-Clipped + HyperNorm (batch size=100)',
         'HyperNet-Clipped + HyperBADGE (batch size=100)',
         'HyperNet-Clipped + HyperBADGE + HyperNorm (batch size=100)',

         'HyperNet-Clipped + Random (batch size=500)',
         'HyperNet-Clipped + HyperNorm (batch size=500)',
         'HyperNet-Clipped + HyperBADGE (batch size=500)',
         'HyperNet-Clipped + HyperBADGE + HyperNorm (batch size=500)',

    ])




def plot_results(experiments, datasetname):
    fig = plt.figure()
    plt.title(datasetname)
    for EXPERIMENT_NAME in experiments:
        try:
            results = np.loadtxt(os.path.join(root, EXPERIMENT_NAME, EXPERIMENT_NAME+'_strategy_performance.txt'))
        except:
            print(f'result not found for {EXPERIMENT_NAME}')
            continue
        plt.plot(results[:,0], results[:,1], label=EXPERIMENT_NAME)
        plt.scatter(results[:,0], results[:,1])
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1.0])
        plt.legend()
        plt.grid('on')
        plt.show()






experiments = [
 'CIFAR10_HyperNethypNetNorm_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethypNetBadgePoinKmeans_100_balldim20_c0.06666666666666667clipr',
 'CIFAR10_HyperNethyperNorm_plus_Rbadge_100_balldim20_c0.06666666666666667clipr',
# ]
# plt.legend([
#          'HyperNet-Clipped + HyperNorm',
#          'HyperNet-Clipped + HyperBADGE',
#          'HyperNet-Clipped + HyperBADGE + HyperNorm'])

# plot_results(experiments, datasetname='CIFAR10, batch size=100')


# 'CIFAR10_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNethyperEmbPoincareKmeans_500_balldim2_c0.06666666666666667_newnormalized',
# 'CIFAR10_HyperNet3hypNetBadge_500_balldim2_c0.06666666666666667_newnormalized',

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
# plt.legend([
#          # 'Net + BADGE',
#          'HyperNet-Clipped + HyperNorm',
#          'HyperNet-Clipped + HyperBADGE',
#          'HyperNet-Clipped + HyperBADGE + HyperNorm'])

plot_results(experiments, datasetname='CIFAR10')
plt.legend([
         # 'Net + BADGE',
         'HyperNet-Clipped + HyperNorm (batch size=100)',
         'HyperNet-Clipped + HyperBADGE (batch size=100)',
         'HyperNet-Clipped + HyperBADGE + HyperNorm (batch size=100)',

         'HyperNet-Clipped + HyperNorm (batch size=500)',
         'HyperNet-Clipped + HyperBADGE (batch size=500)',
         'HyperNet-Clipped + HyperBADGE + HyperNorm (batch size=500)',

          'HyperNet-Clipped + HyperNorm (batch size=1000)',
         'HyperNet-Clipped + HyperBADGE (batch size=1000)',
         'HyperNet-Clipped + HyperBADGE + HyperNorm (batch size=1000)',
         ])


experiments = [
'MNIST_HyperNethypNetNorm_100_balldim20_c0.06666666666666667clipr',



]





