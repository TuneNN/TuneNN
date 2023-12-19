# TuneNN
A transformer-based network model, pitch tracking for musical instruments. 

The timbre of musical notes is the result of various combinations and transformations of harmonic relationships, harmonic strengths and weaknesses, instrument resonant peaks, and structural resonant peaks over time.   

> The online experience based on web audio and tensorflow.js, [See the site here](https://aifasttune.com)  

<img src='./image/tnn.png'   style="width: 800px" > 


- **STFT spectrum**,  the most primitive spectrum, can accurately reflect the harmonic relationships and strengths of harmonics in musical notes. 
- **Bark spectrum**, more accurate than Mel spectrum in accordance with psychoacoustic perception of the human ear, is a nonlinear compression of the STFT spectrum. It belongs to a psychoacoustic abstraction feature that focuses on the harmonic relationships and strengths.   
- **Cepstrum**,  the envelope characteristics of instrument resonant peaks.
- **CQHC**,  MFCC features are designed to address pitch variations in speech. Based on CQT, CQCC can better reflect instrument resonant peaks and structural resonant peaks, while CQHC, using a deconvolution approach, yields more prominent results compared to CQCC. 

**1D value** and **2D time** transformer processed with sliding adjacent windows.  
<p align="center">
	<img src='./image/value.png'   style="width: 350px" > 
</p>
<p align="center">
<img src='./image/time.png'   style="width: 350px" > 
</p>

Specific feature extraction can be referred to in `featureExtract.py`, and the model structure can be referred to in `tuneNN.py`.       

It utilizes the transformer-based tuneNN network model for abstract timbre modeling, supporting tuning for 12+ instrument types.

<p align="center">
  <a target="_blank" href="https://aifasttune.com"><img alt="open in online experience" src="https://img.shields.io/badge/Open%20In%20Online%20Tuner-blue?logo=js&style=for-the-badge&logoColor=green"></a>
</p>

<p align="center">
  	 <img src='./image/fasttune.gif'  style="width: 600px;" >
</p>
