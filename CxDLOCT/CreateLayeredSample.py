import numpy as np
from scipy.ndimage import uniform_filter1d


def CreateLayeredSample(layerPrcts, layerBackScat, layerScat, layersPointSources,
                          nPointSource, objRange, zStart, sampling, layerVar,
                          varType, maxBatchSize=None):
    if maxBatchSize is None:
        maxBatchSize = nPointSource

    batchSize = min(maxBatchSize, nPointSource)
    nBatches = np.ceil(nPointSource / batchSize).astype(int)
    nLayers = len(layerPrcts)

    layerBackScat = np.array([0] + layerBackScat[::-1])
    layerScat = np.array([0] + layerScat)
    layerScatFlip = np.array([0] + layerScat[1:][::-1])
    layersCumPrct = 1 - np.cumsum([0] + layerPrcts[::-1]) / 100
    xVect = np.arange(-objRange[1] / 2 + 1, objRange[1] / 2 + 1) / objRange[1]

    objPos = np.zeros((2, 1, nPointSource))
    objPos[0, 0, :] = objRange[0] * np.random.rand(nPointSource)
    objPos[1, 0, :] = objRange[1] * np.random.rand(nPointSource) - objRange[1] / 2

    layerBounds = np.random.randn(objRange[1], nLayers + 1)
    filterKernel = np.ones((objRange[1], 1))
    layerBounds = uniform_filter1d(layerBounds, size=objRange[1], axis=0)

    linearRamp = xVect[:, None] * np.random.rand(1, nLayers + 1) * 0.25
    layerBounds = (linearRamp + (layerBounds - np.min(layerBounds, axis=0)) / \
                   (np.max(layerBounds, axis=0) - np.min(layerBounds, axis=0)) - 0.5) * \
                  np.hstack([0, np.abs(np.diff((objRange[0] - zStart) * layersCumPrct[:-1])), 0]) * layerVar

    layerBounds = (layerBounds + ((objRange[0] - zStart) * layersCumPrct)) + zStart
    objAmp = np.zeros((1, 1, nPointSource))

    for j in range(nBatches):
        thisBatch = np.unique(np.minimum(((np.arange(batchSize) + (j - 1) * batchSize), nPointSource)))
        approxPos = np.argmax(objPos[1, :, thisBatch] - xVect * objRange[1] < 0, axis=1)
        thisLayer = np.argmax(layerBounds[approxPos, :] - np.squeeze(objPos[0, 0, thisBatch]) <= 0, axis=1)

        layersThickness = np.hstack([np.zeros((objRange[1], 1)), -np.diff(layerBounds, axis=1)])
        objAtt = np.flip(np.cumsum(np.flip(layersThickness, axis=1) * layerScat, axis=1), axis=1) * sampling[0] * 1e3

        objTotalAtt = np.diag(objAtt[approxPos, thisLayer]) + \
            (layerScatFlip[thisLayer] * (np.squeeze(objPos[0, 0, thisBatch]) - np.diag(layerBounds[approxPos, thisLayer]))) * sampling[0] * 1e3

        thisObjAmp = 1e3 * layerScatFlip[thisLayer] * np.exp(-2 * objTotalAtt) * \
            layerBackScat[thisLayer] * (1 + 0.5 * np.random.rand()) * \
            (20 - ((20-1) * ~(np.random.randint(100, size=len(thisBatch)) < 2) & \
                   np.any(nLayers - thisLayer + 2 == layersPointSources)))

        objAmp[0, 0, thisBatch] = np.expand_dims(thisObjAmp, axis=[0, 2])

    objPos = objPos * np.array(sampling)[:, None]

    return objPos, objAmp, layerBounds
