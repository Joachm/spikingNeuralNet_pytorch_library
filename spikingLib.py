import torch as pt
    ## params is a list of values corresponding to
    ## the parameters used in the Izhikevich neurons
    ## the indices are:
    ## 0: k 
    ## 1:resting potential
    ## 2:threshold
    ## 3: C
    ## 4: a
    ## 5: b
    ## 6: peak
    ## 7: reset potential
    ## 8: recovery update

## creates a layer of neurons of size and type specified by parameters
def initializeLayer(params, size):
    membranes = pt.full((size,1), params[1]).cuda()
    recovery = pt.full((size, 1), 0.).cuda()
    spikeTimes = pt.full((size, 1), -1.).long().cuda()
    return membranes, recovery, spikeTimes
    
## create connection matrix between two layers.
## connectivityValue specifies how many connections are non-zero.
## strenghRange specifies the range in which connection strengths can be randomly
## initialized.
def initializeConnections(preSize, postSize, connectivityValue, strengthRange):
    
    IDs = pt.argsort(pt.randint(0,postSize, (preSize, postSize)))[:,:connectivityValue].cuda()

    w = pt.FloatTensor(preSize, connectivityValue).uniform_(strengthRange[0], strengthRange[1]).cuda()

    wMatrix1 = pt.zeros((preSize, postSize)).cuda()

    for i in range(preSize):
        for j in range(connectivityValue):
            wMatrix1[i,IDs[i,j]] = w[i,j]
    return wMatrix1.float()

##all-to-all connections specifically for the last layer
def initializeOutputConnections(preSize, postSize, strengthRange):
    
    wMat = pt.FloatTensor(preSize, postSize).uniform_(strengthRange[0], strengthRange[1]).cuda()
    return wMat 
    

##Finds the IDs of the currently spiking neurons
def findSpiking(spikeTimes, t):
    spiking = spikeTimes==t
    return spiking.reshape(-1)

##Creates an input vector to the postsynaptic layer from the relevant events
def createInputVector(spikingPre, weights):
    
    sPre = spikingPre.float().reshape(1,spikingPre.shape[0])

    inp = pt.matmul(sPre, weights)

    return inp.transpose(0,1) 

    
    ## params is a list of values corresponding to
    ##the parameters used in the Izhikevich neurons
    ## the indices are:
    ## 0: k 
    ## 1:resting potential
    ## 2:threshold
    ## 3: C
    ## 4: a
    ## 5: b
    ## 6: peak
    ## 7: reset potential
    ## 8: recovery update

##Updates the membrane potentials + recovery variables of postsynaptic neurons
def updateNeurons(neurons, recovery, params,  spikeTimes,  inp, currentTime):
    neurons1 = neurons + (params[0]*((neurons - params[1])*(neurons -params[2])- recovery + inp)/params[3])
    
    recovery1 = recovery +  (params[4] * (params[5] * (neurons -params[1] )-recovery  ))
    
    active = neurons1 >= params[6]
    spikeTimes[active==1] = currentTime


    return neurons1, recovery1,  spikeTimes

##Resets the neurons which have just spiked
def reset(neurons, recovery, spikeTimes, spiking, params, currentTime):
    neurons[spiking==1] = params[7]
    recovery[spiking==1] = recovery[spiking==1] + params[8]
    return neurons, recovery

##Determines the members of ar1 that are in ar2
def isIn(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def softMax(x):
    xNorm = x / pt.sum(x)
    return xNorm #pt.exp(xNorm) / pt.sum(pt.exp(xNorm))



def preSTDP1(spikingPre, spikingPost, weights, inp,  LR):
   
    inpSoft = softMax(inp)
    
    spikingPreFloats = spikingPre.float().reshape(spikingPre.shape[0],1)
    
    spikingPostIDs = (spikingPost == True).nonzero()
    nonSpikingPostIDs = (spikingPost == False).nonzero()
    
    nonSpID = nonSpikingPostIDs.flatten()
    spID = spikingPostIDs.flatten()

    wrongPosts = (spikingPost == 0).float()
    wrongPosts = wrongPosts.reshape(wrongPosts.shape[0],1) 
    
    #wrongID  = pt.argsort(inpSoft * wrongPosts, dim=0)[-pt.sum(spikingPost).item():].flatten()
    wrongID  = pt.argmax(inpSoft * wrongPosts)

    posW2Change = weights[:, spID] * spikingPreFloats
    posMask = (posW2Change != 0).float()
    posInpValues = pt.max(inpSoft) - inpSoft[spID]
    posVals = 1 + ((posInpValues * (1 - (posW2Change.transpose(0,1)/1000).transpose(0,1))* LR ) * posMask)



    negW2Change = weights[:, wrongID].reshape(weights[:,wrongID].shape[0],1) *spikingPreFloats 
    wrongInpValues = inpSoft[wrongID].item()
    negVals = 1.- ((negW2Change/1000) * wrongInpValues*LR * posInpValues) 
   
    weights[:, wrongID] *= negVals.flatten()
    weights[:,spID] *= posVals





def preSTDP(spikingPre, spikingPost, weights, inp,  LR):
   
    inpSoft = softMax(inp)
    
    spikingPreFloats = spikingPre.float().reshape(spikingPre.shape[0],1)
    
    spikingPostIDs = (spikingPost == True).nonzero()
    nonSpikingPostIDs = (spikingPost == False).nonzero()
    
    nonSpID = nonSpikingPostIDs.flatten()
    spID = spikingPostIDs.flatten()

    wrongPosts = (spikingPost == 0).float()

    wrongID  = pt.argmax(inpSoft * wrongPosts)

    negW2Change = weights[:, wrongID].reshape(weights[:,wrongID].shape[0],1) *spikingPreFloats 
    wrongInpValues = inpSoft[wrongID].item()
    negVals = 1 - ((negW2Change/1000) * wrongInpValues*LR)

    weights[:, wrongID] *= negVals.flatten()

        

    posW2Change = weights[:, spID] * spikingPreFloats
    posMask = (posW2Change != 0).float()
    posInpValues = 1 - inpSoft[spID]
    posVals = 1 + ((posInpValues * (1 - (posW2Change.transpose(0,1)/1000).transpose(0,1))* LR ) * posMask)
   


    weights[:,spID] *= posVals
   

