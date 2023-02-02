import matplotlib.pyplot as plt
#%% CIFAR10 batch size=10000, Rand and Badge comparison

batch_size_swin_t_CIFAR10 = []
test_accuracy_swin_t_CIFAR10 = []
with open ('swin_t_CIFAR10_10000_badge_scratch_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_swin_t_pre_CIFAR10 = []
test_accuracy_swin_t_pre_CIFAR10 = []
with open ('swin_t_CIFAR10_10000_pretrained_enlarged_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_pre_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_pre_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_pre_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_pre_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_rand_CIFAR10 = []
test_accuracy_rand_CIFAR10 = []
with open ('results/swin_t_CIFAR10_10000_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_CIFAR10, test_accuracy_swin_t_CIFAR10, label='badge')
plt.plot(batch_size_rand_CIFAR10, test_accuracy_rand_CIFAR10, label='Rand')
plt.plot(batch_size_swin_t_pre_CIFAR10, test_accuracy_swin_t_pre_CIFAR10, label='badge_pretrained')

plt.title("CIFAR10 Swin-t 10000")
plt.ylim([0,1.0])
plt.legend()
plt.show()
fig.savefig('CIFAR10_Swin-t_10000.png')


#%% CIFAR10 batch size=1000, Rand and Badge comparison

batch_size_swin_t_CIFAR10 = []
test_accuracy_swin_t_CIFAR10 = []
with open ('results/swin_t_CIFAR10_1000_badge_scratch_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.

batch_size_swin_t_pre_CIFAR10 = []
test_accuracy_swin_t_pre_CIFAR10 = []
with open ('swin_t_CIFAR10_1000_pretrained_enlarged_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_pre_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_pre_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_pre_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_pre_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.

batch_size_rand_CIFAR10 = []
test_accuracy_rand_CIFAR10 = []
with open ('results/swin_t_CIFAR10_1000_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_CIFAR10, test_accuracy_swin_t_CIFAR10, label='badge')
plt.plot(batch_size_rand_CIFAR10, test_accuracy_rand_CIFAR10, label='Rand')
plt.plot(batch_size_swin_t_pre_CIFAR10, test_accuracy_swin_t_pre_CIFAR10, label='badge_pretrained')

plt.title("CIFAR10 Swin-t 1000")
plt.ylim([0,1.0])
plt.legend()
plt.show()
fig.savefig('CIFAR10_Swin-t_1000.png')


#%% CIFAR10 batch size=100, Rand and Badge comparison

batch_size_swin_t_CIFAR10 = []
test_accuracy_swin_t_CIFAR10 = []
with open ('results/swin_t_CIFAR10_100_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_rand_CIFAR10 = []
test_accuracy_rand_CIFAR10 = []
with open ('results/swin_t_CIFAR10_100_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_CIFAR10.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_CIFAR10.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_CIFAR10, test_accuracy_swin_t_CIFAR10, label='badge')
plt.plot(batch_size_rand_CIFAR10, test_accuracy_rand_CIFAR10, label='Rand')
plt.title("CIFAR10 Swin-t 100")
plt.ylim([0,1.0])
plt.legend()
plt.show()
fig.savefig('CIFAR10_Swin-t_100.png')