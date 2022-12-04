import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
from scipy.stats import norm
from scipy import signal
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
import librosa
import soundfile as sf
sys.path.append('/Users/tim-tarek/Desktop/sms-tools-master/software/models')

import utilFunctions as UF
import sineModel as SM
import harmonicModel as HM
from scipy.io.wavfile import write, read

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

#fs = 96000

###################################################### SAMPLE PREPARATION ###########################################################################################################################


def reject_outliers(data, m = 1.1):
	"""
	Remove outliers beyond an absolute distance m from data and return result.
	
	Bannier, B. [benjamin-bannier]. (2013, May 15). Something important when dealing with outliers is that one should try to use estimators as robust as possible. The mean of. [Comment on the online forum post Is there a numpy builtin to reject outliers from a list]. Stack Overflow. https://stackoverflow.com/a/16562028 (accessed: 04.12.2022)

	:param ndarray data: The data from which outliers need to be removed.
	:param float m: Distance m, all values outside of median(data) +/- m will be removed.
	:return: Data with (possibly) removed values.
	:rtype: ndarray
	"""

	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s<m]

def read_frequency_TU_VSL(list_single, index):
	"""
	Return the frequency for sound item from the list_single object.

	:param ndarray list_single: List_single object (created by read_list_single_TU_VSL()) given as input.
	:param int index: Index of sound item from TU-Note Violin sample library
	:return: Frequency of sound item
	:rtype: float
	"""

	return list_single['Hz'][index]

def read_list_single_TU_VSL(path):
	"""
	Load and return the list_single.txt file from path.

	The list_single.txt file contains information string, position, MIDI number, ISO pitch notation, dynamic and frequency in Hz.

	:param string path: Folder path to the list_single.txt file from TU-Note Violin Sample Library.
	:return: contents of list_single as ndarray.
	:rtype: ndarray
	"""

	with open(path, 'r')as file:
		test = []
		file = file.readlines()
		#test = file.copy()
		for line in file:
			test_line = line.replace(" ", "")
			test_line = test_line.replace("\t\t\t", "\t")
			test_line = test_line.replace("\t\t", "\t")
			test = np.append(test, test_line)
			
		list_single = np.genfromtxt(test, delimiter = "\t", comments=None, 
				dtype={'names': (' Fil','Str', 'Pos', 'Midi', 'Iso', 'Dyn', 'Hz'),'formats': ('S20','S1','i4', 'i4', 'S3', 'S2', 'f4')})#,
	return list_single
	

def read_annotations_TU_VSL(path, index):
	"""
	Read the annotation file from the TU-Note Violin Sample Library and return the start and stop points of the sustain part of a sound item in samples.

	:param string path: folder path to annotation file from TU-Note Violin Sample Library.
	:param int index: index of sound item from TU-Note Violin sample library.
	:return float start_sec: Start point of sustain part in samples.
	:return float stop_sec: End point of sustain part in samples.
	"""

	if index <10:
		path_annotations = path+str(0)+str(index)+'.txt' 
	else:
		path_annotations = path+str(index)+'.txt'
	
	annotations = np.loadtxt(path_annotations, dtype={'names': ('time stamp', 'point'),'formats': ('f4', 'S1')})

	start_sec = annotations['time stamp'][1]        #start of sustain part in seconds
	stop_sec = annotations['time stamp'][2]         #end of sustain part in seconds

	return start_sec, stop_sec

def write_parameters(path_soundfile, path_parameters, frequency_f0, index, start_samp, stop_samp, window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, Ns, H, verbose = False):
	"""
	Call harmonic model function from sms-tools and write extracted parameters into files.

	Wrapper function for the sms-tools by Serra. Writes partial amplitude mean, partial amplitude standard deviation, partial frequency standard deviation and mean amplitude of sound item into files. Xavier Serra, “Spectral modeling synthesis tools,” Available at https://www.upf.edu/web/mtg/sms-tools, accessed 29.03.2022, 2013.

	:param string path_soundfile: input folder path, where sound item files are located.
	:param string path_parameters: output folder path for mean amplitude of sample, mean partial amplitudes, and standard deviations of partial amplituds and frequencies.
	:param float frequency_f0: fundamental frequency of sound item.
	:param int index: index of sound item.
	:param int start_samp: Beginning index of segmentation of audio data.
	:param int stop_samp: Ending index of segmentation of audio data.
	:param string window: window type, refer to sms-tools.
	:param int M: window length, refer to sms-tools.
	:param int N: FFT size (must be power of two, must be larger than or equal to M).
	:param float t: spectral peak threshold.
	:param float minSineDur: min duration of sinusoidal tracks.
	:param int nH: number of Harmonics.
	:param float minf0: min f0 frequency.
	:param float maxf0: max f0 frequency.
	:param float f0et: max error in f0 detection.
	:param float harmDevSlope: max deviation of harmonic tracks.
	:param int Ns: size of fft used in synthesis, not used here.
	:param int H: Hop size, must be 1/4 of Ns, if used for sinusoidal synthesis.
	:param bool verbose: if true, prints information on sound item.
	"""

	index = int(index)
	
	if index <10:
		input_path = path_soundfile+str(0)+str(index)+'.wav'
		output_path_amps = os.path.join(path_parameters, "mean_partial_amps/")+str(0)+str(index)+'.txt'
		output_path_mean_amp = os.path.join(path_parameters, "mean_amp/")+str(0)+str(index)+'.txt'
		output_path_std_amps = os.path.join(path_parameters, "std_partial_amps/")+str(0)+str(index)+'.txt'
		output_path_std_freq = os.path.join(path_parameters, "std_partial_freq/")+str(0)+str(index)+'.txt'
	else:
		input_path = path_soundfile+str(index)+'.wav'
		output_path_amps = os.path.join(path_parameters, "mean_partial_amps/")+str(index)+'.txt'
		output_path_mean_amp = os.path.join(path_parameters, "mean_amp/")+str(index)+'.txt'
		output_path_std_amps = os.path.join(path_parameters, "std_partial_amps/")+str(index)+'.txt'
		output_path_std_freq = os.path.join(path_parameters, "std_partial_freq/")+str(index)+'.txt'

    # read input sound
	(fs, x) = read(input_path)

	# one channel only
	if (len(x.shape) !=1):                                   
		raise ValueError("Audio file should be mono")

    #scale down and convert audio into floating point number in range of -1 to 1 (from sms-tools)
	x = np.float32(x)/norm_fact[x.dtype.name]
    
    #save max amplitudein own file
	#max_amp = max(abs(x))
	#with open(output_path_max_amp, "w") as file_m_f:
	#	file_m_f.write(str(max_amp))

	#save mean absolute amplitude in own file
	mean_amp = np.mean(abs(x))
	
	with open(output_path_mean_amp, "w") as file_m_f:
		file_m_f.write(str(mean_amp))


	# cut attack and release part of sample
	x = x[start_samp:stop_samp]

    # compute analysis window
	w = get_window(window, M)

	# extract magnitude information using sms-tools
	hfreq1, hmag1, _ = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    
	#create empty array to vstack mean partial amplitudes 
	means_amp = np.array([])
	std_amp = np.array([])
	std_freq = np.array([])

	for j in range(0,nH):
		trajectory_amp = 10 ** (hmag1[:,j] / 20) #conversion from dB
		mu_a, sigma_a = norm.fit(trajectory_amp)  

		trajectory_freq = hfreq1[:,j]
		trajectory_freq = reject_outliers(trajectory_freq, 1.1)
		mu_f, sigma_f = norm.fit(trajectory_freq)

		if means_amp.size == 0:
			means_amp = np.mean(trajectory_amp)
		else:
			means_amp = np.vstack((means_amp,(np.mean(10 ** (hmag1[:,j] / 20)))))
		
		if std_amp.size == 0:
			std_amp = sigma_a
		else:
			std_amp = np.vstack((std_amp, sigma_a))

		if std_freq.size == 0:
			std_freq = sigma_f
		else:
			std_freq = np.vstack((std_freq, sigma_f))


	with open(output_path_amps, "w") as file_m_f:
		for row in means_amp:
			np.savetxt(file_m_f, row)

	with open(output_path_std_amps, "w") as file_m_f:
		for row in std_amp:
			np.savetxt(file_m_f, row)

	with open(output_path_std_freq, "w") as file_m_f:
		for row in std_freq:
			np.savetxt(file_m_f, row)
    

	if (verbose):
		print("index: ", index, 
		"\nstart sample index: ", start_samp, 
		"\nstop sample index: ", stop_samp, 
		"\nFundamental freq: ", frequency_f0, 
		"\nSampling rate: ", fs,
		"\nmean Amp: ", mean_amp,
		"\npartial Amps: ", means_amp[0:min(3, nH)],
		"\nstd amp: ", std_amp[0:min(3, nH)],
		"\nstd freq: ", std_freq[0:min(3, nH)])

############################################### SAMPLE RATE CONVERSION #################################################################################################################################

def batch_convert_to_44100(file_indices, path_in, path_out):
	"""
	Convert sound items to 44.1kHz.

	:param ndarray file_indices: List of sound item indices, which are to be converted to 44.1kHz.
	:param string path_in: Input path, where sound items are located.
	:param string path_out: Output path, where converted sound items should be placed.
	"""

	for i in file_indices:
    
		sample = int(i)
		print(sample)
		sr = 44100
		if i <10:
			input_path = path_in+str(0)+str(sample)+'.wav'
			output_path = path_out+str(0)+str(sample)+'.wav'
		else:
			input_path = path_in+str(sample)+'.wav'
			output_path = path_out+str(sample)+'.wav'

		audio, s = librosa.load(input_path, sr=sr)
		audio
		sf.write(output_path, audio, 44100)


############################################### SAMPLE CREATION #####################################################################################################################################

def skew_normal_distribution_henze(mu, sigma, theta, size):
	"""
	Draw new values from the skew normal distribution. 

	Implements an algorithm to based on a process outlined in Norbert Henze, “A probabilistic representation of the ‘skew-normal’ distribution,” Scandinavian Journal of Statistics, vol. 13, no. 4, pp. 271–275, 1986.

	:param float mu: location parameter.
	:param float sigma: scale parameter.
	:param float theta: shape, or skewness parameter.
	:param int size: length of return ndarray.
	:return: array of values which have the skew noral distribution.
	:rtype: ndarray
	"""

	U_1 = np.random.normal(0,1, size)
	U_2 = np.random.normal(0,1, size)
	return ((theta * abs(U_1) + U_2)/(np.sqrt(1+theta**2)))*sigma+mu

def interpolation(trajectory, dist_supportpoints, sr):
	"""
	Linear interpolate between support points of parameter trajectory. 

	:param ndarray trajectory: Parameter trajectory made of support points.
	:param int dist_supportpoints: Distance between support points in samples.
	:param int sr: Sampling rate.
	:return: Interpolated parameter trajectory of length len(trajectory) x dist_supportpoints.
	:rtype: ndarray
	"""

	x_trajectory = np.linspace(0, (len(trajectory)-1)*dist_supportpoints/sr, len(trajectory))
	f_int = interp1d(x_trajectory, trajectory, kind='linear')
	x_int = np.linspace(0, (len(trajectory)-1)*dist_supportpoints/sr, len(trajectory)*dist_supportpoints)
	interpolation = f_int(x_int)
	return interpolation

def markov_scaled_normal_synthesis(list_indices, 
									list_single, 
									path_amp,
									path_vol,
									path_output,
									alpha_f0 = 0.001,
									alpha_amp= 0.001,
									sigma_f0 = 0.004, 
									sigma_amp = 0.02, 
									num_samples = 600,
									len_support_points = 512,
									sr = 44100,
									num_partials = 80
									):
	"""
	Synthesize violin sound using the scaled normal Markovian synthesis.

	Synthesis starts with a given value and draws following values from a normal distribution, where the location parameter is scaled between a target value and the preceding value.

	:param list list_indices: List of indices of sound items to be recreated.
	:param ndarray list_single: List_single object (created by read_list_single_TU_VSL()) given as input.
	:param string path_amp: Path to the partial amplitudes extracted by write_parameters().
	:param string path_vol: Path to the overall loudness extracted by write_parameters().
	:param string path_output: Path to where synthesized sound items should be saved.
	:param float alpha_f0: Scaling the influence of target value or preceding value on location parameter for frequency trajectory.
	:param float alpha_amp: Scaling the influence of target value or preceding value on location parameter for amplitude trajectory.
	:param float sigma_f0: Standard deviation for the frequency trajectory.
	:param float sigma_amp: Standard deviation for the amplitude trajectory.
	:param int num_samples: Number of sample points. Needs to be greater than or equal to 1.
	:param int len_support_points: Length of support points between sample points.
	:param int sr: Sampling rate. Can be anything, but shoulde be 44100 in order to be analyzed with sms-tools.
	:param int num_partials: Number of partials to be created for the output sound.
	"""

	for samp_ind in tqdm(list_indices):
		sample_index = samp_ind

		target = list_single['Hz'][sample_index] 

		if sample_index <10:
			amp_path = path_amp+str(0)+str(sample_index)+'.txt'
			vol_path = path_vol+str(0)+str(sample_index)+'.txt'
			
		else:
			amp_path = path_amp+str(sample_index)+'.txt'
			vol_path = path_vol+str(sample_index)+'.txt'

		Amp = np.loadtxt(amp_path)
		vol = np.loadtxt(vol_path)


		#weighting influence of target / preceding value 
		beta_f0 = 1-alpha_f0
		beta_amp = 1-alpha_amp

		final_sound = np.zeros(len_support_points*num_samples)

		t = np.linspace(0, (num_samples*len_support_points-1)/sr, num_samples*len_support_points) +1

		for i in range(0, num_partials):
			results = []
			# create individual jitter partial trajectory
			stored_value = target*(i+1)

			for j in range(0, num_samples):
				new_value = np.random.normal((alpha_f0*target*(i+1)+beta_f0*stored_value), sigma_f0*(i+1)/1, 1)
				results = np.append(results, new_value)
				stored_value = new_value

			trajectory_f = results


			trajectory_f_int = interpolation(trajectory_f, len_support_points, sr)

			#create phase 
			phase = 2*np.pi*(target*(i+1)+(trajectory_f_int-target*(i+1))/(t))*t

			#create sinusoid
			sound = np.sin(phase)

			# Amplitude Envelope by same procedure
			target_amp = 1        
			stored_value = target_amp
			results = []
			for k in range(0, num_samples):
				new_value = np.random.normal((alpha_amp*target_amp+beta_amp*stored_value), sigma_amp, 1)
				results = np.append(results, new_value)
				stored_value = new_value
			trajectory_amp = results
			trajectory_amp_int = interpolation(trajectory_amp, len_support_points, sr)
			sound = sound * trajectory_amp_int

			#add sinusoids together and scale amplitude of partials
			final_sound = final_sound + sound*Amp[i]

		#final sound with fade in envelope
		envelope = np.ones(len(final_sound))
		fade_len_ms = 0.3
		if len(final_sound)>fade_len_ms*2*sr:
			#print("adding fade envelope")
			for m in range(0, int(fade_len_ms*sr)): #for 100 ms
				envelope[m]= m/(fade_len_ms*sr)
				envelope[-m] = m/(fade_len_ms*sr)
		final_sound = final_sound * envelope

		#Normalization    
		#final_sound_norm = final_sound/max(abs(final_sound)) #peak normalization 
		final_sound_norm = final_sound/np.mean(abs(final_sound)) #mean absolute normalization

		#Loudness matching with loudness from source sound
		final_sound_norm = final_sound_norm*vol
		
		#Create string containing all parameters
		value_string = "_af0_"+str(alpha_f0)+"_bf0_"+str(beta_f0)+"_sf0_"+str(sigma_f0)+"_aAmp_"+str(alpha_amp)+"_bAmp_"+str(beta_amp)+"_sAmp_"+str(sigma_amp)
		
		if sample_index <10:
			path_write = path_output+str(0)+str(sample_index)+str(value_string)+".wav"
		else:
			path_write = path_output+str(sample_index)+str(value_string)+".wav"
		write(path_write, sr, final_sound_norm.astype(np.float32))




def markov_skew_normal_synthesis(list_indices,
								list_single, 
								path_amp,
								path_vol,
								path_output,
								gamma_f0 = 1,
								gamma_amp= 0.8,
								sigma_f0 = 0.008, 
								sigma_amp = 0.03, 
								num_samples = 600,
								len_support_points = 512,
								sr = 44100,
								num_partials = 80
								):
	"""
	Synthesize violin sound using the skew normal Markovian synthesis.

	Synthesis starts with a given value and draws following values from a skew normal distribution, where the skew parameter is calculated by the distance between target value and preceding value multiplied by a scaling factor gamma. Based on the formula for skew normal distribution by Henze. Norbert Henze, “A probabilistic representation of the ‘skew-normal’ distribution,” Scandinavian Journal of Statistics, vol. 13, no. 4, pp. 271–275, 1986.

	:param list list_indices: List of indices of sound items to be recreated.
	:param ndarray list_single: List_single object (created by read_list_single_TU_VSL()) given as input.
	:param string path_amp: Path to the partial amplitudes extracted by write_parameters().
	:param string path_vol: Path to the overall loudness extracted by write_parameters().
	:param string path_output: Path to where synthesized sound items should be saved.
	:param float gamma_f0: Scaling the influence of target value or preceding value on skewness parameter for frequency trajectory.
	:param float alpha_amp: Scaling the influence of target value or preceding value on skewness parameter for amplitude trajectory.
	:param float sigma_f0: Standard deviation for the frequency trajectory.
	:param float sigma_amp: Standard deviation for the amplitude trajectory.
	:param int num_samples: Number of sample points. Needs to be greater than or equal to 1.
	:param int len_support_points: Length of support points between sample points.
	:param int sr: Sampling rate. Can be anything, but shoulde be 44100 in order to be analyzed with sms-tools.
	:param int num_partials: Number of partials to be created for the output sound.
	"""

	for samp_ind in tqdm(list_indices):
    
		sample_index = samp_ind

		target = list_single['Hz'][sample_index] 

		if sample_index <10:
			amp_path = path_amp+str(0)+str(sample_index)+'.txt'
			vol_path = path_vol+str(0)+str(sample_index)+'.txt'
		else:
			amp_path = path_amp+str(sample_index)+'.txt'
			vol_path = path_vol+str(sample_index)+'.txt'

		Amp = np.loadtxt(amp_path)
		vol = np.loadtxt(vol_path)

		final_sound = np.zeros(len_support_points*num_samples)

		t = np.linspace(0, (num_samples*len_support_points-1)/sr, num_samples*len_support_points)+1

		for i in range(0,num_partials):

			# create individual frequency trajectory for each partial
			results_skew = []
			stored_value = target*(i+1)
			for j in range(0, num_samples):
				new_value = skew_normal_distribution_henze((stored_value), sigma_f0*(i+1), -gamma_f0*(stored_value-target*(i+1)),1)
				results_skew = np.append(results_skew, new_value)
				stored_value = new_value
			trajectory_f = results_skew

			# interpolate
			trajectory_f_int = interpolation(trajectory_f, len_support_points, sr)

			#ax_freq.plot(trajectory_f_int)


			#create phase 
			phase = 2*np.pi*(target*(i+1)+(trajectory_f_int-target*(i+1))/(t))*t

			#create sinusoids
			sound = np.sin(phase)

			# create individual amplitude trajectory for each partial
			target_amp = 1        
			stored_value = target_amp
			results_skew = []
			for k in range(0, num_samples):
				new_value = skew_normal_distribution_henze((stored_value), sigma_amp, -gamma_amp*(stored_value-target_amp),1)
				results_skew = np.append(results_skew, new_value)
				stored_value = new_value
			trajectory_amp = results_skew
			trajectory_amp_int = interpolation(trajectory_amp, len_support_points, sr)
			sound = sound * trajectory_amp_int  

			#ax_amp.plot(trajectory_amp_int + 2* i)

			#add sinusoids together and scale amplitude of partials
			final_sound = final_sound + sound*Amp[i]

		#final sound with fade in envelope
		envelope = np.ones(len(final_sound))
		fade_len_ms = 0.3
		if len(final_sound)>fade_len_ms*2*sr:
			#adding fade envelope
			for m in range(0, int(fade_len_ms*sr)): #for 100 ms
				envelope[m]= m/(fade_len_ms*sr)
				envelope[-m] = m/(fade_len_ms*sr)
		final_sound = final_sound * envelope


		#Normalization    
		#final_sound_norm = final_sound/max(abs(final_sound)) #peak normalization 
		final_sound_norm = final_sound/np.mean(abs(final_sound)) #mean absolute normalization
		
		#Loudness matching with loudness from source sound
		final_sound_norm = final_sound_norm*vol

		#Create string containing all parameters
		value_string = "_gf0_"+str(gamma_f0)+"_sf0_"+str(sigma_f0)+"_gAmp_"+str(gamma_amp)+"_sAmp_"+str(sigma_amp)
		
		if sample_index <10:
			path_write = path_output+str(0)+str(sample_index)+str(value_string)+".wav"
		else:
			path_write = path_output+str(sample_index)+str(value_string)+".wav"


		write(path_write, sr, final_sound_norm.astype(np.float32))

    

############################################# SAMPLE VISUALIZATION #################################################################################################################################

def plot_waveform(index, path_in, path_annotations, path_save, mode, dyn, save=False):
	"""
	Plot and optionally save the waveform of the original or the synthesized soundfiles.

	Currently expects files to start with the index of the sound item followed by an underscore, and to have only one file per sound item. Plotting of original waveforms shows full sample and sustain part, but only when using mode = "original".

	:param int index: Index of sound item.
	:param string path_in: Path to the soundfiles (Original, or skew or scaled synthesized).
	:param string path_annotations: Path to the annotation files for the attack, decay, sustain and release segmentation.
	:param string path_save: Path to where output figures should be saved.
	:param string mode: Origin of sound file. Options are "original", "skew" and "scaled".
	:param string dyn: Simply adds the dynamic marker (pp, mp, mf, ff) to the file name.
	:param bool save: Saves resulting plot at location specified by path_save.
	"""

	if (mode == "original"):
		if index <10:
			input_path = path_in+str(0)+str(index)+'.wav'

		else:
			input_path = path_in+str(index)+'.wav'
	else:
		list_folder = os.listdir(path_in)

		if index <10:
			filename = [filename for filename in list_folder if filename.startswith("0"+ str(index)+"_")]
			input_path = path_in+filename[0]

		else:
			filename = [filename for filename in list_folder if filename.startswith(str(index)+"_")]
			input_path = path_in+filename[0]

	(fs, x) = read(input_path)

	x = np.float32(x)/norm_fact[x.dtype.name]

	t = np.arange(0, len(x))/fs 

	if (mode == "original"):
		start_sec, stop_sec = read_annotations_TU_VSL(path_annotations, index)
		y = x[int(start_sec*fs):int(stop_sec*fs)]
		t2 =  np.arange(0, len(y))/fs + start_sec

	fig = plt.figure(figsize=(12, 5))
	fig.tight_layout()

	if (mode == "original"):
		plt.plot(t, x, color = "lightgray", label = "Full sample")
		plt.plot(t2, y, color = "gray", label = "Sustain part")
	else:
		plt.plot(t, x, color = "gray")


	plt.xlabel("Duration in seconds", fontsize = 20)
	plt.ylabel("Amplitude", fontsize = 20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	if (mode == "original"):
		plt.legend(fontsize = 20)
	plt.tight_layout()

	if (save):
		plt.savefig(path_save+str(index)+"_"+str(dyn)+"_"+str(mode))

def plot_FFT(index, path_in, path_annotations, path_save, mode, dyn, save=False):
    """
	Plot and optionally save the Fourier transform of the original or the synthesized soundfiles.

	Currently expects files to start with the index of the sound item followed by an underscore, and to have only one file per sound item. Plotting of original FFT uses only sustain part, but only when using mode = "original".

	:param int index: Index of sound item.
	:param string path_in: Path to the soundfiles (Original, or skew or scaled synthesized).
	:param string path_annotations: Path to the annotation files for the attack, decay, sustain and release segmentation.
	:param string path_save: Path to where output figures should be saved.
	:param string mode: Origin of sound file. Options are "original", "skew" and "scaled".
	:param string dyn: Simply adds the dynamic marker (pp, mp, mf, ff) to the file name.
	:param bool save: Saves resulting plot at location specified by path_save.
	"""
	
    if (mode == "original"):
        if index <10:
            input_path = path_in+str(0)+str(index)+'.wav'

        else:
            input_path = path_in+str(index)+'.wav'
    else:
        list_folder = os.listdir(path_in)

        if index <10:
            filename = [filename for filename in list_folder if filename.startswith("0"+ str(index)+"_")]
            input_path = path_in+filename[0]

        else:
            filename = [filename for filename in list_folder if filename.startswith(str(index)+"_")]
            input_path = path_in+filename[0]

    (fs, x) = read(input_path)

    x = np.float32(x)/norm_fact[x.dtype.name]

    t = np.arange(0, len(x))/fs 

    if (mode == "original"):
        start_sec, stop_sec = read_annotations_TU_VSL(path_annotations, index)
        x = x[int(start_sec*fs):int(stop_sec*fs)]
        
    X = abs(np.fft.fft(x)/len(x))
    X = X[range(int(len(x)/2))] 
    X_dbfs = 20 * np.log10(X / 1.0)  
    frequencies = np.arange(int(len(x)/2))*fs/len(x)

    fig = plt.figure(figsize=(12, 5))
    fig.tight_layout()

    plt.plot(frequencies, X_dbfs, color = "gray")


    plt.xlabel("Frequency in Hz", fontsize = 20)
    plt.ylabel("Amplitude in dB", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(10, 20000)
    plt.ylim(-120,0)
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    

    if (save):
        plt.savefig(path_save+"_FFT_"+str(index)+"_"+str(dyn)+"_"+str(mode))


def plot_spectrogram(index, path_in, path_annotations, path_save, mode, dyn, save=False):
	"""
	Plot and optionally save the spectrogram of the original or the synthesized soundfiles.

	Basically just a wrapper for the librosa.display.specshow() function. Currently expects files to start with the index of the sound item followed by an underscore, and to have only one file per sound item. Currently needs path to the 44.1k original sound items instead of the 96k original sound items. Plotting of original FFT uses only sustain part, but only when using mode = "original".

	:param int index: Index of sound item.
	:param string path_in: Path to the soundfiles (Original, or skew or scaled synthesized).
	:param string path_annotations: Path to the annotation files for the attack, decay, sustain and release segmentation.
	:param string path_save: Path to where output figures should be saved.
	:param string mode: Origin of sound file. Options are "original", "skew" and "scaled".
	:param string dyn: 
	:param bool save: 
	"""

	if (mode == "original"):
		if index <10:
			input_path = path_in+str(0)+str(index)+'.wav'

		else:
			input_path = path_in+str(index)+'.wav'
	else:
		list_folder = os.listdir(path_in)

		if index <10:
			filename = [filename for filename in list_folder if filename.startswith("0"+ str(index)+"_")]
			input_path = path_in+filename[0]

		else:
			filename = [filename for filename in list_folder if filename.startswith(str(index)+"_")]
			input_path = path_in+filename[0]

	(fs, x) = read(input_path)

	x = np.float32(x)/norm_fact[x.dtype.name]

	t = np.arange(0, len(x))/fs 

	if (mode == "original"):
		start_sec, stop_sec = read_annotations_TU_VSL(path_annotations, index)
		x = x[int(start_sec*fs):int(stop_sec*fs)]
		
	STFT = librosa.stft(x)  
	S_db = librosa.amplitude_to_db(np.abs(STFT), ref=np.max)



	fig = plt.figure(figsize=(12,5))
	fig.tight_layout()
	librosa.display.specshow(S_db, x_axis='time', y_axis='linear')
	cbar = plt.colorbar(format="%+2.f dB")
	cbar.ax.tick_params(labelsize=20) 
	plt.xlabel("Time in sec", fontsize = 20)
	plt.ylabel("Frequency in Hz", fontsize = 20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.tight_layout()
	if (save):
		plt.savefig(path_save+"_SPECT_"+str(index)+"_"+str(dyn)+"_"+str(mode))