Installation
------------
The code for Continuous State Modeling for Statistical Spectral Synthesis can be downloaded from `Github`_.

.. _Github: https://github.com/yamitarek/CSM4SS

The analysis functions require the `SMS-Tools`_ to be installed.

.. _SMS-Tools: https://github.com/MTG/sms-tools

It solely makes use of the samples from the `TU-Note Violin Sample Library`_ and requires the directory structure to be left as is.

.. _TU-Note Violin Sample Library: https://depositonce.tu-berlin.de/items/f81ba73c-4d9b-48de-9fbc-31cb03d5b9bc

Analysis
--------
The results of the analysis part are contained in *../CSM4SS/extracted_parameters*. If you like to refine these, you can follow this guide.

The github repository contains the python notebook 0_sample_preparation.ipynb.
You can use it to prepare and analyze the TU-Note Violin samples.
Make sure to change the following folder paths to the correct directories:

- ``path_frequency`` should link to the *../TU-Note_Violin/File_Lists/list_Single.txt* file.
- ``path_annotations`` should link to *../TU-Note_Violin/Segments/SingleSounds/SampLib_DPA_*. It is important to leave the path ending with the underscore, as a counting variable will be added to the path later in order to access the single sound items.
- ``path_soundfile_96k`` should link to *../TU-Note_Violin/WAV/SingleSounds/BuK/SampLib_BuK_*. Again, let the path end with an underscore.

You can leave the other folder paths unchanged.

You can use the function :py:func:`CSM_functions.batch_convert_to_44100`.
While the analysis functions in this library can deal with other sampling rates, the standalone GUI version of the SMS-Tools throws an error when trying to read in files with a higher sampling rate.
The GUI can help with testing out the parameters for the analysis functions. 

Next you can extract the relevant meta-information on the sound files. 
Informations like the dynamics and frequencies are contained in the *list_single.txt* file, which you can load using :py:func:`CSM_functions.read_list_single_TU_VSL`.
Using the :py:func:`CSM_functions.read_frequency_TU_VSL` function, you can extract the frequency from the object.
Likewise, you can extract the sustain segmentation time stamps from the *SampLib_DPA_XX* files using the :py:func:`CSM_functions.read_annotations_TU_VSL` function.

Finally, you can analyze and write the parameters into files using the :py:func:`CSM_functions.write_parameters` function.
Caution: Calling this function will overwrite the existing files contained in *../CSM4SS/extracted_parameters*.
You can batch extract parameters by using a list of integers (maximum: from 01 to 336). However, not all values work for all samples.
You can refine them by reusing the function with different parameters to overwrite the current files.


Synthesis
---------
To synthesize sound items using Markovian spectral synthesis you can use the 1_sample_creation.ipynb.

Make sure to change the following folder paths to the correct directories:
- ``path_frequency`` should link to the *../TU-Note_Violin/File_Lists/list_Single.txt* file.
- ``path_annotations`` should link to *../TU-Note_Violin/Segments/SingleSounds/SampLib_DPA_*. It is important to leave the path ending with the underscore.

Afterwards, create the list_single object by calling :py:func:`CSM_functions.read_list_single_TU_VSL`.
Finally, you can use :py:func:`CSM_functions.markov_scaled_normal_synthesis` and :py:func:`CSM_functions.markov_skew_normal_synthesis` to create samples.
By using a list for the *list_indices* argument you can batch create samples following the name structure for frequency and dynamics from the TU-Note Violin Sample library.


