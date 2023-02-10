import matplotlib.pyplot as plt
#%% MNIST batch size=10000, Rand and Badge comparison

batch_size_swin_t_MNIST = []
test_accuracy_swin_t_MNIST = []
with open ('swin_t_MNIST_10000_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_rand_MNIST = []
test_accuracy_rand_MNIST = []
with open ('swin_t_MNIST_10000_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_MNIST, test_accuracy_swin_t_MNIST, label='badge')
plt.plot(batch_size_rand_MNIST, test_accuracy_rand_MNIST, label='Rand')
plt.title("MNIST Swin-t 10000")
plt.ylim([0,1.0])
plt.legend()
plt.show()
fig.savefig('MNIST_Swin-t_10000.png')



#%% MNIST batch size=100, Rand and Badge comparison

batch_size_swin_t_MNIST = []
test_accuracy_swin_t_MNIST = []
with open ('swin_t_MNIST_1000_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_hypUMAP_MNIST = []
test_accuracy_hypUMAP_MNIST = []
with open ('swin_t_MNIST_1000_hypUMAP_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_hypUMAP_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_hypUMAP_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_hypUMAP_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_hypUMAP_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))

batch_size_rand_MNIST = []
test_accuracy_rand_MNIST = []
with open ('swin_t_MNIST_1000_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_MNIST, test_accuracy_swin_t_MNIST, label='badge')
plt.plot(batch_size_rand_MNIST, test_accuracy_rand_MNIST, label='Rand')
plt.plot(batch_size_hypUMAP_MNIST, test_accuracy_hypUMAP_MNIST, label='HypUmap')
plt.title("MNIST Swin-t 1000")
plt.ylim([0.5,1.0])
plt.legend()
plt.show()
fig.savefig('MNIST_Swin-t_1000_2.png')



#%% MNIST batch size=100, Rand and Badge comparison

batch_size_swin_t_MNIST = []
test_accuracy_swin_t_MNIST = []
with open ('results/swin_t_MNIST_100_badge_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_swin_t_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_swin_t_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))                  # and print the string.


batch_size_rand_MNIST = []
test_accuracy_rand_MNIST = []
with open ('results/swin_t_MNIST_100_Rand_log.txt', 'rt') as myfile:  # Open lorem.txt for reading
    for myline in myfile:              # For each line, read to a string,
        if 'testing accuracy' in myline:
            if 't'in myline.split(' ')[0][0:5]:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:3]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:3]), float(myline.split(' ')[-1]))
            else:
                batch_size_rand_MNIST.append(int(myline.split(' ')[0][0:5]))
                test_accuracy_rand_MNIST.append(float(myline.split(' ')[-1]))
                print(int(myline.split(' ')[0][0:5]), float(myline.split(' ')[-1]))


fig = plt.figure()
plt.plot(batch_size_swin_t_MNIST, test_accuracy_swin_t_MNIST, label='badge')
plt.plot(batch_size_rand_MNIST, test_accuracy_rand_MNIST, label='Rand')
plt.title("MNIST Swin-t 100")
plt.ylim([0,1.0])
plt.legend()
plt.show()
fig.savefig('MNIST_Swin-t_100.png')