# import packages
import os
from multiprocessing.spawn import freeze_support
from torchviz import make_dot

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pandas import read_pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
import torchviz
import pickle
from sklearn.preprocessing import scale
import hiddenlayer as hl

SMALLP = 0.3

GLOBALP = 0.2
LATENT_SPACE_DIM = 20
NUM_EPOCHS = 400000
LR = 0.01
BATCH_SIZE =128
NOISE_LEVEL =0.1
i=0
def create_df():
    genres=pd.read_pickle('genres_3.pickle')
    users_vector = read_pickle('full_user_ff_3.pickle')
    user_vector_a=users_vector.toarray()
    genres_names = pd.Series(['mean_rating', 'std_rating', 'genre_dist_mean', 'gengre_dist_std'])
    ccg = pd.Series(genres.columns)
    agg = pd.Series(genres.columns)
    for col,i in zip(ccg,range(ccg.size)):
        agg[i]=col+" Avg rat"
    cols = genres_names.append(ccg).append(agg)
    aa = pd.DataFrame(user_vector_a, columns=cols)
    return aa ,users_vector

class Denoising_AE(nn.Module):
    def __init__(self):
        super(Denoising_AE, self).__init__()
        self.dropout=nn.Dropout(p=GLOBALP)
        self.dropout_weaker=nn.Dropout(p=SMALLP)

        # encoder
        self.enc0 = nn.Linear(in_features=58, out_features=30)
        self.enc1 = nn.Linear(in_features=30, out_features=30)
        self.enc2 = nn.Linear(in_features=30, out_features=25)
        self.enc3 = nn.Linear(in_features=25, out_features=25)
        self.enc4 = nn.Linear(in_features=25, out_features=25)
        self.enc5 = nn.Linear(in_features=25, out_features=20)
        self.enc6 = nn.Linear(in_features=20, out_features=20)
        self.enc7 = nn.Linear(in_features=20, out_features=20)
        self.enc8 = nn.Linear(in_features=20, out_features=20)
        self.enc9 = nn.Linear(in_features=20, out_features=20)
        self.enc10 = nn.Linear(in_features=20, out_features=20)
        self.enc11 = nn.Linear(in_features=20, out_features=20)
        self.enc12 = nn.Linear(in_features=20, out_features=20)
        self.enc13 = nn.Linear(in_features=20, out_features=20)
        self.enc14 = nn.Linear(in_features=20, out_features=20)
        self.enc15 = nn.Linear(in_features=20, out_features=20)
        self.enc16 = nn.Linear(in_features=20, out_features=20)
        self.enc17 = nn.Linear(in_features=20, out_features=20)
        self.enc18 = nn.Linear(in_features=20, out_features=20)
        self.enc18 = nn.Linear(in_features=20, out_features=20)
        self.enc19 = nn.Linear(in_features=20, out_features=20)
        self.enc20 = nn.Linear(in_features=20, out_features=20)
        self.enc21 = nn.Linear(in_features=20, out_features=15)
        self.enc22 = nn.Linear(in_features=15, out_features=LATENT_SPACE_DIM)


        self.dec0 = nn.Linear(in_features=LATENT_SPACE_DIM, out_features=15)
        self.dec1 = nn.Linear(in_features=15, out_features=20)
        self.dec2 = nn.Linear(in_features=20, out_features=20)
        self.dec3 = nn.Linear(in_features=20, out_features=20)
        self.dec4 = nn.Linear(in_features=20, out_features=20)
        self.dec5 = nn.Linear(in_features=20, out_features=20)
        self.dec6 = nn.Linear(in_features=20, out_features=20)
        self.dec7 = nn.Linear(in_features=20, out_features=20)
        self.dec8 = nn.Linear(in_features=20, out_features=20)
        self.dec9 = nn.Linear(in_features=20, out_features=20)
        self.dec10 = nn.Linear(in_features=20, out_features=20)
        self.dec11= nn.Linear(in_features=20, out_features=20)
        self.dec12 = nn.Linear(in_features=20, out_features=20)
        self.dec13 = nn.Linear(in_features=20, out_features=20)
        self.dec14 = nn.Linear(in_features=20, out_features=20)
        self.dec15 = nn.Linear(in_features=20, out_features=20)
        self.dec16 = nn.Linear(in_features=20, out_features=25)
        self.dec17 = nn.Linear(in_features=25, out_features=25)
        self.dec18 = nn.Linear(in_features=25, out_features=25)
        self.dec19 = nn.Linear(in_features=25, out_features=25)
        self.dec20 = nn.Linear(in_features=25, out_features=30)
        self.dec21 = nn.Linear(in_features=30, out_features=30)
        self.dec22 = nn.Linear(in_features=30, out_features=58)

    def encoder(self,x):
        x = F.relu(self.enc0(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc1(x))
        x =self.dropout(F.relu(self.enc2(x)))
        x =  (F.relu(self.enc3(x)))
        x =self.dropout(F.relu(self.enc4(x)))
        x = F.relu(self.enc5(x))
        x=self.dropout_weaker(x)


        x = F.relu(self.enc6(x))
        x=self.dropout_weaker(x)

        x=F.relu(self.enc7(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc8(x))
        x=self.dropout_weaker(x)

        x = (self.enc9(x))
        x=self.dropout_weaker(x)
        x=torch.sigmoid(x)

        x = F.relu(self.enc10(x))
        x =self.dropout(self.enc11(x))
        x = F.relu(self.enc12(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc13(x))
        x=self.dropout_weaker(x)

        x = (self.enc14(x))
        # x=torch.sigmoid(x)

        x =self.dropout(F.relu(self.enc15(x)))

        x = F.relu(self.enc16(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc17(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc18(x))
        x=self.dropout_weaker(x)
        # x=torch.sigmoid(x)

        x = F.relu(self.enc19(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc20(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc21(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.enc22(x))

        return x

    def decoder(self,x):
        x = F.relu(self.dec0(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec1(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec2(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec3(x))
        x=self.dropout_weaker(x)
        # x=torch.sigmoid(x)
        x = (self.dec4(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec5(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec6(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec7(x))
        x=self.dropout(self.dec8(x))
        x = F.relu(self.dec9(x))
        x=self.dropout_weaker(x)
        # x=torch.sigmoid(x)
        x = F.relu(self.dec10(x))
        x=self.dropout_weaker(x)


        x =  F.relu(self.dec11(x))
        x=self.dropout_weaker(x)


        x = F.relu(self.dec12(x))
        x=self.dropout(self.dec13(x))


        x = F.relu(self.dec14(x))

        x =F.relu(self.dec15(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec16(x))
        x=self.dropout(F.relu(self.dec17(x)))

        x = F.relu(self.dec18(x))
        x=self.dropout_weaker(x)
        x=torch.sigmoid(x)

        x = F.relu(self.dec19(x))
        x=self.dropout_weaker(x)

        x = F.relu(self.dec20(x))
        x=(self.dec21(x))
        x=self.dropout_weaker(x)


        x = F.relu(self.dec22(x))

        x=torch.sigmoid(x)
        return x

    def forward(self, x):
        global i
        if self.training:
            x=x+torch.tensor(np.random.normal(loc=0, scale=NOISE_LEVEL, size=x.shape).astype(np.float32))

        latent_var=self.encoder(x)
        decoded_var=self.decoder(latent_var)
        i += 1

        if i % (BATCH_SIZE) == 0:

            print("latent varaible {}".format(pd.DataFrame(latent_var.detach().clone().numpy()[:2,:])))


        return decoded_var

def test_pred(net, testloader,epoch,criterion):
    net.eval()
    run_loss=0
    for user in testloader:
        outputs = net(user)
        loss=criterion(outputs,user)
        run_loss+=loss.item()
    loss=run_loss/len(testloader)

    print('Epoch {} of {}, Test Loss: {:.3f}'.format(
        epoch + 1, NUM_EPOCHS, loss))
    return loss

def train(net, trainloader,test_loader, NUM_EPOCHS,optimizer,criterion):
    train_loss = []
    test_loss=[]
    net.train()
    for epoch in range(1,NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            outputs = net(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            # Additional Info when using cuda

            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print("Train start")
        los_t=test_pred(net,test_loader,epoch,criterion)
        test_loss.append(los_t)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))

        if (epoch)%3==0:
            sns.lineplot(x=np.arange(epoch), y=(test_loss), label="test").set_title(
                "Current epoch : " + str(epoch) +"\n noise level :"+str(NOISE_LEVEL) +
                " , Learning Rate : "+str(LR)+", Batch size :" +str(BATCH_SIZE)+"\n Loss test : " +str(los_t)[:4] +" ,Loss train "+ str(running_loss)[:4])
            sns.lineplot(x=np.arange(epoch), y=(train_loss), label="train")
            plt.legend()
            plt.show()
        if epoch%10==0:
            torch.save(net.state_dict(), "weights_ae\weights_curr"+str(epoch))

        # if los_t<0.05:
        #     torch.save(net.state_dict(), " weights_ae\weights_curr_loss_" + str(los_t))
        #


    return train_loss

from sklearn.preprocessing import maxabs_scale

def data_prep():
    sns.set()
    users, user_vector = create_df()
    np.set_printoptions(precision=3,suppress=True)
    ninty_per = int((users.shape[0]) * 0.9)
    users.iloc[:,31:]=users.iloc[:,31:]*5
    users.iloc[:,2]=0
    users=maxabs_scale(users)
    print(users[:3,:10])
    # Maybe try binary corss entropy
    users_vector_train = users[:ninty_per]
    users_vector_test = users[ninty_per:, :]
    print(ninty_per)
    aa = np.array(users_vector_train).astype(np.float32)
    bb = np.array(users_vector_test).astype(np.float32)
    train_loader = DataLoader(aa, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(bb, batch_size=BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return  users, test_loader, train_loader


def main():

    users, test_loader, train_loader = data_prep()
    net = Denoising_AE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    train(net,train_loader,test_loader,NUM_EPOCHS,optimizer=optimizer,criterion=criterion)
    torch.save(net.state_dict(),"weights_curr")



if __name__=='__main__':
    main()