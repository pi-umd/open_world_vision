import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms



def train_model(dataloaders, clsModel, loss_CrossEntropy, 
                optimizerW, schedulerW, 
                num_epochs=50, work_dir='./', device='cpu', freqShow=40):
    
    log_filename = os.path.join(work_dir, 'train.log')    
    since = time.time()
    best_loss = float('inf')
    best_acc = 0.
    best_perClassAcc = 0.0
    
    phaseList = list(dataloaders.keys())
    if 'test' in phaseList:
        phaseList.remove('test')    

    phaseList.remove('train')
    phaseList = ['train'] + phaseList
    
    
    for epoch in range(num_epochs):        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()

        # Each epoch has a training and validation phase
        for phase in phaseList:
            print(phase)
            if phase !='train':
                predList = np.array([])
                grndList = np.array([])
                predList_cur = np.array([])
                grndList_cur = np.array([])
                
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                schedulerW.step()                
                clsModel.train()
            else:
                clsModel.eval()  # Set model to training mode  
              
            running_loss_CE = 0.0
            running_loss = 0.0
            running_acc = 0.0
            countSmpl = 0.
            
            # Iterate over data.
            iterCount, sampleCount = 0, 0
            for sample in dataloaders[phase]:                
                imageList224, labelList = sample
                imageList224 = imageList224.to(device)
                labelList = labelList.type(torch.long).view(-1).to(device)

                # zero the parameter gradients
                optimizerW.zero_grad()
                
                # forward
                # track history if only in train
                loss = 0
                
                with torch.set_grad_enabled(phase=='train'):
                    logits = clsModel(imageList224)
                    softmaxScores = logits.softmax(dim=1)

                    predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)                  
                    accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
                    accRate = (accRate==0).type(torch.float).mean()

                    if phase != 'train':
                        predList = np.concatenate((predList, predLabel.cpu().numpy()))
                        grndList = np.concatenate((grndList, labelList.cpu().numpy()))

                    error_CE = loss_CrossEntropy(logits, labelList)
                    error = error_CE

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        error.backward()
                        optimizerW.step()

                # statistics  
                iterCount += 1
                sampleCount += labelList.size(0)
                running_acc += accRate*labelList.size(0) 
                running_loss_CE += error_CE.item() * labelList.size(0) 
                running_loss = running_loss_CE
                
                print2screen_avgLoss = running_loss / sampleCount
                print2screen_avgLoss_CE = running_loss_CE / sampleCount
                print2screen_avgAccRate = running_acc / sampleCount
                       
                del loss                
                
                if iterCount%freqShow==0:
                    print('\t{}/{} loss:{:.3f}, loss_CE:{:.3f}, acc-full:{:.5f}'.
                          format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, 
                                print2screen_avgLoss_CE, print2screen_avgAccRate))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss:{:.3f}, loss_CE:{:.3f}, acc-full:{:.5f}\n'.
                             format( iterCount, len(dataloaders[phase]), print2screen_avgLoss, 
                                print2screen_avgLoss_CE, print2screen_avgAccRate))
                    fn.close()
                    
            epoch_error = print2screen_avgLoss      
                    
            print('\tloss: {:.6f}, acc-full: {:.6f}'.format(
                epoch_error, print2screen_avgAccRate))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f}, acc-full: {:.6f}\n'.format(
                epoch_error, print2screen_avgAccRate))
            fn.close()
            
            # deep copy the model
            path_to_save_param = os.path.join(work_dir, 'epoch-{}_classifier.paramOnly'.format(epoch+1))
            torch.save(clsModel.state_dict(), path_to_save_param)
            
            if phase=='val' or phase=='test':
                confMat = sklearn.metrics.confusion_matrix(grndList, predList)                
                # normalize the confusion matrix
                a = confMat.sum(axis=1).reshape((-1,1))
                confMat = confMat / a
                curPerClassAcc = 0
                for i in range(confMat.shape[0]):
                    curPerClassAcc += confMat[i,i]
                curPerClassAcc /= confMat.shape[0]
                
                
            if (phase=='val' or phase=='test') and curPerClassAcc>best_perClassAcc: #epoch_loss<best_loss:            
                best_loss = epoch_error
                best_acc = print2screen_avgAccRate
                best_perClassAcc = curPerClassAcc

                path_to_save_param = os.path.join(work_dir, 'best_classifier.paramOnly')
                torch.save(clsModel.state_dict(), path_to_save_param)
                
                file_to_note_bestModel = os.path.join(work_dir,'note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.5f}, acc-full{:.5f}.\n'.format(
                    epoch+1, best_loss, best_perClassAcc))
                fn.close()

                
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
    
    return True
