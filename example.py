import pickle
from pyNeurons import *
from loadMNISTData import *
np.random.seed(0)
pt.manual_seed(0)
pt.backends.cudnn.deterministic = True
pt.backends.cudnn.benchmark = False

##Load data
trainImages, trainLabels = loadMNIST("train", "/home/Desktop/MNIST")

testImages = pt.from_numpy(trainImages[50000:]).float().cuda()
testLabels = pt.from_numpy(trainLabels[50000:]).long().cuda()

trainImages = pt.from_numpy(trainImages[:50000]).float().cuda()
trainLabels = pt.from_numpy(trainLabels[:50000]).long().cuda()

params = pt.FloatTensor([0.7, -60., -40., 100., 0.03, -2., -40., -50., 100.]).cuda()


visSize = 784
visNeu, visRec, visST= initializeLayer(params, visSize)

encSize = 4000
encNeu, encRec, encST= initializeLayer(params, visSize)

outSize = 10
outNeu, outRec, outST = initializeLayer(params, outSize)


### Make vis to enc connections
connect = 40
w1 = initializeConnections(visSize, encSize, connect, [400, 400])


### Make enc to out connections
w2 = initializeOutputConnections(encSize, outSize, [600,400])



numberOf = 50000
span = 10
time = 0
epochs = 20


results = pt.zeros(epochs).cuda()

for epoch in range(epochs):
    print('epoch: ', epoch) 
    order = np.arange(numberOf)
    np.random.shuffle(order)
    order = pt.from_numpy(order)
    trainLabels = trainLabels[order]
    trainImages = trainImages[order]

    for im in range(numberOf):
        visNeu, visRec, visST = initializeLayer(params, visSize)
        encNeu, encRec, encST= initializeLayer(params, encSize)

        for t in range(span):
                        
            visInput = trainImages[im].reshape(visSize,1)*6
            
            visNeu, visRec, visST = updateNeurons(visNeu, visRec, params, visST, visInput, time)

            spikingVis = findSpiking(visST,time)

            encInput = createInputVector(spikingVis, w1)
            
            encNeu, encRec, encST = updateNeurons(encNeu, encRec, params, encST, encInput, time)

            spikingEnc = findSpiking(encST, time)


            outInput = createInputVector(spikingEnc, w2)

            if pt.sum(spikingEnc) > 0:
                outST[trainLabels[im]] = time

                spikingOut = findSpiking(outST, time)
                preSTDP1(spikingEnc, spikingOut, w2,outInput, 10)

            w2 = pt.clamp(w2, 0,1000)
            visNeu, visRec = reset(visNeu, visRec, visST,spikingVis, params, time)
            encNeu, encRec = reset(encNeu, encRec, encST,spikingEnc, params, time)
            time +=1

    pickle.dump((w1, w2), open('latestWeights.pickle', 'wb'))

