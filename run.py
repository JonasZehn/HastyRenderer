import subprocess
import json
from pathlib import Path
import pathlib

import os

scenes = ['assets/buddha_mat', 'assets/CornellBox' , 'assets/glass_full4' , 'assets/m-box', 'assets/materialTest' , 'assets/metalRing' , 'assets/prism'  ]

unidirectional_executable = './build-debugoptimized/PathTracingViewer.exe';
stochastic_photon_executable = './build-debugoptimized/PhotonMappingViewer.exe';
executable = stochastic_photon_executable
inputFile = './renderJob.json'
outputFolder = Path('./output')

with open(inputFile) as f:
    inputJSON = json.load(f)

print(inputJSON)

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

for scene in scenes:
    inputJSON['scene'] = str(Path(__file__).parent.absolute() / (scene + '.json') ) 
    
    sceneBaseName = os.path.basename(scene)
    sceneFolder = outputFolder / sceneBaseName
    
    if not os.path.exists(sceneFolder):
        os.makedirs(sceneFolder)
    
    outputFile = sceneFolder / 'renderJob.json'
    with open(outputFile, 'w') as f:
        json.dump(inputJSON, f)

    subprocess.check_call([str(executable), str(outputFile), str(sceneFolder)])
