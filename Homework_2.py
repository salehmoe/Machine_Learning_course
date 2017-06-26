#
import numpy as np

def ReadInput(sampleLow,sampleHigh):
    Data= np.loadtxt('housing.data')
    Input=Data[sampleLow:sampleHigh,0:13]
    Input1=np.c_[ np.ones(sampleHigh-sampleLow), Input]
    for features in range(0,14):
        maximum=Input1.max(axis=0)[features]
        #print (maximum)
        if(maximum!=0):                     # normalizing
            Input1[:,features]=Input1[:,features]/maximum               # normalizing
        means=Input1.mean(axis=0)[features]                         #recentering
        if(features!=0):
            Input1[:,features]=Input1[:,features]-means                 #recentering
            stds=Input1.std(axis=0)[features]                           #recentering
            if(stds!=0):
                Input1[:,features]=Input1[:,features]/stds
#    print (Input1)
    return Input1

def ReadOutput(sampleLow,sampleHigh):
    Data= np.loadtxt('housing.data')
    Output=Data[sampleLow:sampleHigh,13]
    return Output



def Weight(sampleLength):                                           ####### Weight vector function  #############
    Data= np.loadtxt('housing.data')
    Input=Data[0:sampleLength,0:13]
    Output=Data[0:sampleLength,13]
    Input1=np.c_[ np.ones(sampleLength), Input]
    #print(Input1)
    for features in range(0,14):
        maximum=Input1.max(axis=0)[features]
        #print (maximum)
        if(maximum!=0):                     # normalizing
            Input1[:,features]=Input1[:,features]/maximum               # normalizing
        means=Input1.mean(axis=0)[features]                         #recentering
        if(features!=0):
            Input1[:,features]=Input1[:,features]-means                 #recentering
            stds=Input1.std(axis=0)[features]                           #recentering
            if(stds!=0):
                Input1[:,features]=Input1[:,features]/stds                  #recentering
    if(np.linalg.det(np.dot(np.transpose(Input1),Input1))==0):
        print("not invertable")
    Weight=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Input1)
    ,Input1)),
    np.transpose(Input1)),Output) ### weight vector
    #print (Weight)
    return Weight



def Predict(weightv,case):                                          ########## Prediction function    ##########
    output=np.dot(weightv,np.transpose(case))
    return output


def MSE(outputreal,outputpredicted):                                  ########## MSE function       ##########
    sum=0
    for i in range(0,np.size(outputreal)):
        sum=sum+(outputreal[i]-outputpredicted[i])*(outputreal[i]-
        outputpredicted[i])
    return sum/np.size(outputreal)




switcher=np.array([200,300,400])


for i in range(0,3):
    sampledim=switcher[i]
############ trainning set MSE
    weightvector=Weight(sampledim)
    InputTrain=ReadInput(0,sampledim)
    predictedoutput=np.empty([sampledim,1])
    for i in range(0,sampledim):
        predictedoutput[i]=Predict(weightvector,InputTrain[i])

    OutputTrain=ReadOutput(0,sampledim)
    MSETrain=MSE(OutputTrain,predictedoutput)
    print ('MSE for training for training set of ' + str(sampledim) +
    '=' + str(MSETrain))

    ########### Test Set MSE
    InputTest=ReadInput(sampledim,505)
    predictedoutputTest=np.empty([505-sampledim,1])
    for i in range(0,505-sampledim):
        predictedoutputTest[i]=Predict(weightvector,InputTest[i])
    OutputTest=ReadOutput(sampledim,505)
    MSETest=MSE(OutputTest,predictedoutputTest)
    print ('MSE for test for training set of ' + str(sampledim) +
    '=' + str(MSETest))
