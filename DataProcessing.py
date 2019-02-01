import os
import glob
import tensorflow as tf
import time
import numpy as np
from scipy.io import wavfile
import csv
from operator import eq
#from itertools import 

class Params(object):
	def __init__(self):
		self.Names2Label={	"air_conditioner":0, "car_horn":1,
							"children_playing":2,"dog_bark":3,
							"drilling":4,"engine_idling":5,
							"gun_shot":6,"jackhammer":7,
							"siren":8,"street_music":9
						}
		self.WavFileDirectory='./URBAN-SED_v2.0.0/audio/train/'
		self.TxtAnnotationDirectory='./URBAN-SED_v2.0.0/annotations/train/'
		self.NumClasses=10
		self.TotalEpochs=20
		self.SignalLength=441000
		self.WindowSize=11025
	
	def FetchAnnotation(self,Name):
		"""
		We take the annotation data and generate a one hot label for each 
		element in the sound vector.

		For a sound of type 	 [aaaabbbbaaa]
		we have the annotation  [[00001111000],
								 [11110000111]]
		which is a psuedo one hot vector annotationn
		"""
		Annotation=np.zeros((self.SignalLength,self.NumClasses))
		with open(Name) as f:
			reader = csv.reader(f, delimiter="\t")
			AnnotationCSV = list(reader)
		for category in AnnotationCSV:
			Annotation[int(44100*float(category[0])):int(44100*float(category[1])),self.Names2Label[category[2]]]=1
		return Annotation

	def FetchSignal(self,Name):
		SamplingFrequency, Data = wavfile.read(Name)
		return Data

	def FetchInputsAndLabels(self):
		WaveFilesList=[]
		AnnotationFileList=[]

		for file in glob.glob(self.WavFileDirectory+"*.wav"):
			WaveFilesList.append(file)

		for file in glob.glob(self.TxtAnnotationDirectory+"*.txt"):
			AnnotationFileList.append(file)

		WaveFilesList.sort()	
		AnnotationFileList.sort()

		
		for WavFileName,AnnotationFileName in zip(WaveFilesList,AnnotationFileList):
			"""

			Some reshaping op

			"""
			WaveArray=self.FetchSignal(WavFileName)
			LabelArray=self.FetchAnnotation(Name=AnnotationFileName)

			DynamicBatchSize=np.array(WaveArray.shape)//self.WindowSize
			
			BatchedWaveFile=np.zeros((DynamicBatchSize[0],self.WindowSize))
			BatchedLabelFile=np.zeros((DynamicBatchSize[0],self.WindowSize,self.NumClasses))

			for i in range(DynamicBatchSize):
				BatchedWaveFile[i,:]=WaveArray[i*self.WindowSize:(i+1)*self.WindowSize]
				BatchedLabelFile[i,:,:]=LabelArray[i*self.WindowSize:(i+1)*self.WindowSize,:]
			yield BatchedWaveFile,BatchedLabelFile
			
		


			
				



	
