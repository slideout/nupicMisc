#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import importlib
import sys, os
import csv
import datetime
import pprint
import pickle

from nupic.swarming import permutations_runner
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.model import Model
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager


SWARM_CONFIG = {
    "includedFields": [
        {
            "fieldName": "timestamp",
            "fieldType": "datetime"
        },
        {
            "fieldName": "fileCount",
            "fieldType": "int",
            "maxValue": 100,
            "minValue": 0
        }
    ],
    "streamDef": {
        "info": "fileCount",
        "version": 1,
        "streams": [
            {
                "info": "fileData.csv",
                "source": "file://fileData.csv",
                "columns": [ "*" ]
            }
        ]
    },
    "inferenceType": "TemporalAnomaly",
    #"inferenceType": "TemporalMultiStep",
    "inferenceArgs": {
        "predictionSteps": [ 1 ],
        "predictedField": "fileCount"
    },
    "iterationCount": -1,
    "swarmSize": "large"
}
SWARM_TEMP_FOLDER = "./swarmTemp"
MODEL_PARAMS = "modelParams"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

_METRIC_SPECS = (
    MetricSpec(
        field='fileCount',
        metric='multiStep',
        inferenceElement='multiStepBestPredictions',
        params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='fileCount',
        metric='trivial',
        inferenceElement='prediction',
        params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='fileCount',
        metric='multiStep',
        inferenceElement='multiStepBestPredictions',
        params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
    MetricSpec(
        field='fileCount',
        metric='trivial',
        inferenceElement='prediction',
        params={'errorMetric': 'altMAPE', 'window': 1000, 'steps': 1}),
    )



def swarm():
    # swarm using the config and get the result params
    params = permutations_runner.runWithConfig(
        SWARM_CONFIG,
        { 'maxWorkers': 8, 'overwrite': True },
        outDir=SWARM_TEMP_FOLDER,
        permWorkDir=SWARM_TEMP_FOLDER,
        verbosity=0
    )

    # save these so they can be loaded by the learn / model create
    pp = pprint.PrettyPrinter( indent=2 )
    formatted = pp.pformat( params )
    with open( MODEL_PARAMS + ".py", 'wb' ) as paramsFile:
        paramsFile.write( 'MODEL_PARAMS = \\\n%s' % formatted )


def train():
    # load the params module
    try:
        modelModule = importlib.import_module( MODEL_PARAMS ).MODEL_PARAMS
    except ImportError:
        raise Exception( "FU buddy, no model params found" )

    # create the model
    model = ModelFactory.create(modelModule)
    model.enableInference({"predictedField": "fileCount"})

    # get the data into a csv reader
    inputFile = open("fileData.csv", "rb")
    csvReader = csv.reader(inputFile)

    # create the output csv writer, the results
    outputFile = open("fileData_train.csv", "wb" )
    csvWriter = csv.writer( outputFile )
    csvWriter.writerow( ['timestamp','fileCount','predictedFileCount'] )

    counter = 0
    for trainCounter in range(50):
        # reset the file to position zero again
        inputFile.seek(0)
        csvReader.next()  # skip the header rows
        csvReader.next()
        csvReader.next()

        for row in csvReader:
            counter += 1

            timestamp = datetime.datetime.strptime( row[0], DATE_FORMAT )
            fileCount = int(row[1])

            result = model.run({ "timestamp": timestamp, "fileCount": fileCount })
            p1 = result.inferences["multiStepBestPredictions"][1]
            a = result.inferences["anomalyScore"]
            csvWriter.writerow([timestamp, fileCount, ' NP:'+str(p1), ' A:'+str(a)])

            if counter % 100 == 0:
                print "pass %i, %i records loaded" % (trainCounter, counter)

    inputFile.close()
    outputFile.close()

    model.save( os.path.abspath( 'modelSave' ) )

def test(args):

    inputName = ''
    if 'good' in args:
        inputName = 'fileDataGOOD.csv'
    elif 'bad' in args:
        inputName = 'fileDataBAD.csv'
    else:
        print 'Yer killin me smalls, specify a test (good, bad)'
        return

    # load the previously trained model from disk
    model = Model.load( os.path.abspath( 'modelSave' ))

    # setup the metrics handling
    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(), model.getInferenceType())

    inputFile = open(inputName, "rb")
    csvReader = csv.reader(inputFile)
    csvReader.next()  # skip the header rows
    csvReader.next()
    csvReader.next()

    outputFile = open("fileData_test.csv", "wb" )
    csvWriter = csv.writer( outputFile )
    csvWriter.writerow( ['timestamp','fileCount','nextPredictedFileCount','currentAnomaly'] )

    for row in csvReader:
        timestamp = datetime.datetime.strptime( row[0], DATE_FORMAT )
        fileCount = int(row[1])

        # run the data
        result = model.run({ "timestamp": timestamp, "fileCount": fileCount })

        # get the raw inference data
        p1 = result.inferences["multiStepBestPredictions"][1]
        a = result.inferences["anomalyScore"]
        csvWriter.writerow([timestamp, fileCount, ' NP:'+str(p1), ' A:'+str(a)])
        print( '%s %i NP:%s A:%s' % (timestamp, fileCount, str(p1), str(a)) )

        # get the metrics data, over-time avarages and such
        # not sure how to interpret these yet... tbd
        result.metrics = metricsManager.update( result )
        # print result.metrics
        #print result.metrics[
        #    "multiStepBestPredictions:multiStep:"
        #    "errorMetric='altMAPE':steps=1:window=1000:"
        #    "field=kw_energy_consumption"]


    inputFile.close()
    outputFile.close()

    # save it again, if the recently learned data needs to be updated
    # model.save( os.path.abspath( 'modelSave' ) )


if __name__ == "__main__":

  args = sys.argv[1:]
  if "swarm" in args:
      swarm()

  if "train" in args:
      train()

  if "test" in args:
      test( args )
