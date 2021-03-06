import csv
from matplotlib import pyplot as plt
import matplotlib as mpl

"""
Auxliary scripts to generate plots by reading data from tensorboard
Not part of the mail code
"""

def read_values_csv(filename):
    loss =[]
    epochs = []
    with open (filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        header = True
        for row in csv_reader:
            print(row , "\n")
            if not header:
                loss.append(abs(float((row[2]))))
            else:
                header = False
    # returns list of all loss scalars
    for epoch in enumerate(loss):
        epochs.append(epoch[0])
    
    return loss, epochs

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

figure = plt.figure()
import pdb
pdb.set_trace()
gen_loss, epochs_loss_gen = read_values_csv("plots/data/4.1/run-Jul16_18-17-06_gpu20-37alldata_no_LR_sch-tag-Valid_Accuracy.csv")
gen_loss, epochs_loss_gen = read_values_csv("plots/data/4.1/run-Jul16_18-17-06_gpu20-37alldata_no_LR_sch-tag-Valid_Accuracy.csv")
gen_loss, epochs_loss_gen = read_values_csv("plots/data/4.1/run-Jul16_18-17-06_gpu20-37alldata_no_LR_sch-tag-Valid_Accuracy.csv")
discr_loss, epochs_loss_discr = read_values_csv("run-.-tag-Discriminator_loss.csv")


plt.plot(epochs_loss_gen[::100], gen_loss[::100], color="red", label="Generator Loss")
plt.plot(epochs_loss_discr[::100], discr_loss[::100], color="green", label="Discriminator Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Discriminator and Generator Loss after 20k epochs WGAN")
plt.legend()


figure.savefig('loss.png')

plt.show()   
