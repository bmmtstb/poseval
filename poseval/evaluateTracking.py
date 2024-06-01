import numpy as np
import json
import os
import sys

from . import eval_helpers
from .eval_helpers import Joint
import motmetrics as mm


def computeMetrics(gtFramesAll, motAll, outputDir, bSaveAll, bSaveSeq):

    assert(len(gtFramesAll) == len(motAll))

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    # intermediate metrics
    metricsMidNames = ['num_misses', 'num_switches', 'num_false_positives', 'num_objects', 'num_detections']

    # final metrics computed from intermediate metrics
    metricsFinNames = ['mota', 'motp', 'pre', 'rec']

    # initialize intermediate metrics
    metricsMidAll = {}
    for name in metricsMidNames:
        metricsMidAll[name] = np.zeros([1, nJoints])
    metricsMidAll['sumD'] = np.zeros([1, nJoints])

    # initialize final metrics
    metricsFinAll = {}

    # create metrics
    mh = mm.metrics.create()

    imgidxfirst = 0
    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    nSeq = len(seqidxsUniq)

    # initialize per-sequence metrics
    metricsSeqAll = {}
    for si in range(nSeq):
        metricsSeqAll[si] = {}
        for name in metricsFinNames:
            metricsSeqAll[si][name] = np.zeros([1, nJoints+1])

    names = Joint().name
    names['15'] = 'total'

    for si in range(nSeq):
        print("seqidx: %d/%d" % (si+1, nSeq))

        # init per-joint metrics accumulator
        accAll = {}
        for i in range(nJoints):
            accAll[i] = mm.MOTAccumulator(auto_id=True)

        # extract frames IDs for the sequence
        imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
        imgidxs = imgidxs[:-1].copy()
        seqName = gtFramesAll[imgidxs[0, 0]]["seq_name"]
        print(seqName)
        # create an accumulator that will be updated during each frame
        # iterate over frames
        for j in range(len(imgidxs)):
            imgidx = imgidxs[j, 0]
            # iterate over joints
            for i in range(nJoints):
                # GT tracking ID
                trackidxGT = motAll[imgidx][i]["trackidxGT"]
                # prediction tracking ID
                trackidxPr = motAll[imgidx][i]["trackidxPr"]
                # distance GT <-> pred part to compute MOT metrics
                # 'NaN' means force no match
                dist = motAll[imgidx][i]["dist"]
                # Call update once per frame
                accAll[i].update(
                    trackidxGT,                 # Ground truth objects in this frame
                    trackidxPr,                 # Detector hypotheses in this frame
                    dist                        # Distances from objects to hypotheses
                )

        # compute intermediate metrics per joint per sequence
        for i in range(nJoints):
            metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')
            for name in metricsMidNames:
                metricsMidAll[name][0, i] += metricsMid[name]
            s = accAll[i].events['D'].sum()
            if np.isnan(s):
                s = 0
            metricsMidAll['sumD'][0, i] += s

            if bSaveSeq:
                # reuse metrics per joint per sequence
                # compute final metrics per sequence
                numObj = metricsMid['num_objects'] if metricsMid['num_objects'] > 0 else np.nan
                numFP = metricsMid['num_false_positives']
                metricsSeqAll[si]['mota'][0, i] = 100.*(1. - 1.*(
                    (
                        metricsMid['num_misses'] + metricsMid['num_switches'] + numFP
                    ) / numObj))
                numDet = metricsMid['num_detections']
                metricsSeqAll[si]['motp'][0, i] = 0.0 if numDet == 0 else 100.*(1. - (1. * s / numDet))
                totalDet = numFP+numDet if numFP+numDet > 0 else np.nan
                metricsSeqAll[si]['pre'][0, i]  = 100.*(1.*numDet / totalDet)
                metricsSeqAll[si]['rec'][0, i]  = 100.*(1.*numDet / numObj)

        if bSaveSeq:
            # average metrics over all joints per sequence
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['mota'][0, :nJoints]))
            metricsSeqAll[si]['mota'][0, nJoints] = metricsSeqAll[si]['mota'][0, idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['motp'][0, :nJoints]))
            metricsSeqAll[si]['motp'][0, nJoints] = metricsSeqAll[si]['motp'][0, idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['pre'][0, :nJoints]))
            metricsSeqAll[si]['pre'][0, nJoints]  = metricsSeqAll[si]['pre'] [0, idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['rec'][0, :nJoints]))
            metricsSeqAll[si]['rec'][0, nJoints]  = metricsSeqAll[si]['rec'] [0, idxs].mean()

            metricsSeq = metricsSeqAll[si].copy()
            metricsSeq['mota'] = metricsSeq['mota'].flatten().tolist()
            metricsSeq['motp'] = metricsSeq['motp'].flatten().tolist()
            metricsSeq['pre'] = metricsSeq['pre'].flatten().tolist()
            metricsSeq['rec'] = metricsSeq['rec'].flatten().tolist()
            metricsSeq['names'] = names

            filename = os.path.join(outputDir, f'/{seqName}_MOT_metrics.json')
            print('saving results to', filename)
            eval_helpers.writeJson(metricsSeq, filename)

    # compute final metrics per joint for all sequences
    numFP = metricsMidAll['num_false_positives']
    numDet = metricsMidAll['num_detections']
    s = metricsMidAll['sumD']

    numObj = metricsMidAll['num_objects']
    numObj[numObj <= 0] = np.nan

    # total number of detections
    totalDet = numFP + numDet
    totalDet[totalDet <= 0] = np.nan

    # MOTA
    metricsFinAll['mota'] = 100. * (1. - (
            (
                    metricsMidAll['num_misses'] + metricsMidAll['num_switches'] + numFP
            ).astype(np.float64) / numObj.astype(np.float64)))
            # ).astype(np.float64) / totalDet.astype(np.float64)))

    # MOTP
    # use fancy np.divide, to make sure there is no division by nan or zero and set all those values to zero.
    motp = np.zeros_like(s, dtype=np.float64)
    np.divide(s, numDet, out=motp, where=~((numDet == 0) | np.isnan(s)))
    metricsFinAll['motp'] = 100. * (1. - motp)

    # precision and recall
    metricsFinAll['pre'] = 100. * (numDet.astype(np.float64) / totalDet.astype(np.float64))
    metricsFinAll['rec'] = 100. * (numDet.astype(np.float64) / numObj.astype(np.float64))


    # add one more dimension to be able to save the mean
    metricsFinAll['mota'] = np.pad(metricsFinAll['mota'], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    metricsFinAll['motp'] = np.pad(metricsFinAll['motp'], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    metricsFinAll['pre'] = np.pad(metricsFinAll['pre'], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    metricsFinAll['rec'] = np.pad(metricsFinAll['rec'], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    # average metrics over all joints over all sequences
    idxs = np.argwhere(~np.isnan(metricsFinAll['mota'][0, :nJoints]))
    metricsFinAll['mota'][0, nJoints] = metricsFinAll['mota'][0, idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['motp'][0, :nJoints]))
    metricsFinAll['motp'][0, nJoints] = metricsFinAll['motp'][0, idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['pre'][0, :nJoints]))
    metricsFinAll['pre'][0, nJoints]  = metricsFinAll['pre'] [0, idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['rec'][0, :nJoints]))
    metricsFinAll['rec'][0, nJoints]  = metricsFinAll['rec'] [0, idxs].mean()

    if (bSaveAll):
        metricsFin = metricsFinAll.copy()
        metricsFin['mota'] = metricsFin['mota'].flatten().tolist()
        metricsFin['motp'] = metricsFin['motp'].flatten().tolist()
        metricsFin['pre'] = metricsFin['pre'].flatten().tolist()
        metricsFin['rec'] = metricsFin['rec'].flatten().tolist()
        metricsFin['names'] = names

        filename = os.path.join(outputDir, './total_MOT_metrics.json')
        print('saving results to', filename)
        eval_helpers.writeJson(metricsFin, filename)

    return metricsFinAll


def evaluateTracking(gtFramesAll, prFramesAll, outputDir, saveAll=True, saveSeq=False):

    distThresh = 0.5
    # assign predicted poses to GT poses
    _, _, _, motAll = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute MOT metrics per part
    metricsAll = computeMetrics(gtFramesAll, motAll, outputDir, saveAll, saveSeq)

    return metricsAll
