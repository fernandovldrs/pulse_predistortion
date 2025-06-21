"""
This script is meant to extract the IIR filters raw data obtained by running the flux-pi scope experiment.

IMPORTANT: This script assumes that you have already ran hdf5_extraction.py on the raw data file.

Sample run (change /User/kyle/Documents to whatever path the fast-flux-line repo is in):
python3 flux_pi_scope_analysis.py /Users/kyle/Documents/fast-flux-line/data/090729_cut.h5
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import dlsim, convolve

from inverse_filtering import iterative_iir_fitting, fir_fitting
from process_flux_pi_scope import run_convert_flux_pi_plot
from pulses import square_pulse
from utils import DataType


### Input Options ##
PATH = f""

WAVEFORM_TYPE = DataType.LINE_RESPONSE  # Can be either PI_SCOPE or LINE_RESPONSE
FOLDER = PATH + ("pi_scope\\" if WAVEFORM_TYPE == DataType.PI_SCOPE 
                              else "line_response\\")
FILE = "20250621_231354_converted_pi_scope_lr"
INPUT_WAVEFORM_PATH = FOLDER + FILE + ".npz"

### Filter Saving Options ###
SAVE_FILTERS = 0
FILTER_SAVE_LOCATION = "filters/pi_scope_filter_20250622_IIR_A.pickle"

SAMPLING_PERIOD = 1e-9
### IIR Filtering Options ###
DO_IIR = 1
DISP = 0
IIR_SAMPLE_POINTS = [
                     [3201 + DISP, 5000 + DISP],
                     [201 + DISP, 4000 + DISP],
                     [650 + DISP, 3200 + DISP],
                     ]

# Define bounds for parameters a, b, tau of each IIR fitting. Default if None.
IIR_PARAM_BOUNDS = [((0, 0, 0), (2.0, np.inf, np.inf)),
                    ((0, -np.inf, 0), (5.0, 0, np.inf)),
                    ((0, -np.inf, 0), (5.0, 0, np.inf)),
                    # ((0, -np.inf, 0), (2.0, 0, np.inf)),
                    #((0, -np.inf, 0), (10.0, 0, np.inf)),
                    ]

IIR_PARAM_GUESS = [(0.0, 1, 3000e-9),
                   (0.5, -1, 1000e-9),
                   (0.5, -1, 1000e-9),
                #    (0.5, -1, 1000e-9),
                   ]

PLOT_IIR_FITS = 1 # Plot sampling and exponential fit for each IIR filter
PLOT_FREQUENCY_FITS = 0 # Show fit of qubit spec for each timestep

### FIR Filtering Options ###
DO_FIR = 0
FIR_FILTER_NUM = 0
FIR_FILTERS = [None] * FIR_FILTER_NUM
FIR_SAMPLE_POINTS = [[8*4 + DISP, 70*4 + DISP],]  # in ns
STEP_DELAYS = [
    17*4 + DISP - 8*4 - DISP,
]  # where you want the target to be, relative to the sample points
PLOT_FIR_SAMPLES = 1
PLOT_INTERMEDIATE_FIR_RESULTS = 1
FIR_REGULARIZER_WEIGHT = 5
EXTEND_FIR_CORRECTION = 200

###  Other Plotting Options ###
PLOT_NORMALIZED_DATA = 0
PLOT_PREDISTORTED_PULSE = 1

if WAVEFORM_TYPE == DataType.PI_SCOPE:
    input_wavefrom = run_convert_flux_pi_plot(
        file_path=INPUT_WAVEFORM_PATH, plot_final=True, plot_individual_fits=PLOT_FREQUENCY_FITS,
    )
    input_wavefrom = np.concatenate(([input_wavefrom[0]]*DISP, input_wavefrom), axis = 0)
    np.savez(PATH + "line_response\\20250621_231354_converted_pi_scope_lr.npz", input_wavefrom)
else:
    data_file = np.load(INPUT_WAVEFORM_PATH)
    input_wavefrom = data_file["arr_0"]
    a1 = (np.array([input_wavefrom[0]]*DISP))
    a2 = (input_wavefrom)

    input_wavefrom = np.concatenate((a1, a2), axis = 0)
    print(input_wavefrom.shape, "AAAAAAA")
    
time_points = np.arange(len(input_wavefrom)) * SAMPLING_PERIOD

print("---------- ZEROING & NORMALIZING WAVEFORM ----------")
zero_point = input_wavefrom[0]  # lets set the start of the waveform to be 0
waveform_shifted = input_wavefrom - zero_point
max_amp = max(waveform_shifted)
normalized_waveform = waveform_shifted / max_amp

print("--------- GENERATING FILTERS ----------")

if DO_IIR:
    print("---------- PERFORMING IIR FITTING ----------")
    print(normalized_waveform.shape, time_points.shape )
    iir_filters, final_correction = iterative_iir_fitting(
        normalized_waveform,
        IIR_SAMPLE_POINTS,
        IIR_PARAM_BOUNDS,
        IIR_PARAM_GUESS,
        time_points,
        dt=SAMPLING_PERIOD,
        plot_fits=PLOT_IIR_FITS,
    )
else:
    final_correction = normalized_waveform
    iir_filters = []

if DO_FIR:
    print("----------PERFORMING FIR FITTING----------")
    # estimate step function
    curr_correction = final_correction
    for i in range(FIR_FILTER_NUM):
        step_delay = STEP_DELAYS[i]
        start = FIR_SAMPLE_POINTS[i][0]
        end = FIR_SAMPLE_POINTS[i][1]
        sample = curr_correction[start:end]
        estimated_deltafunction = np.zeros(len(sample))
        estimated_deltafunction[step_delay] = 1
        
        fir_filter = fir_fitting(
            step_response=sample,
            expxected_response=estimated_deltafunction,
            alpha=FIR_REGULARIZER_WEIGHT,
        )
        print(f"FIR filter{i}: {fir_filter.x}")
        FIR_FILTERS[i] = fir_filter.x
        ### Extend corrected pulse in time to account for FIR smudging
        extend_curr_correction = np.concatenate(([curr_correction[0]]
                                                 *EXTEND_FIR_CORRECTION,
                                                 curr_correction), axis = 0)
        corrected_pulse = convolve(extend_curr_correction, 
                                   fir_filter.x, mode="same")
        
        prev_time_points = time_points
        time_points = np.concatenate((time_points, 
                              [time_points[-1] + 1e-9*x 
                               for x in range(1, EXTEND_FIR_CORRECTION + 1)]), 
                              axis = 0)
        if PLOT_INTERMEDIATE_FIR_RESULTS:
            fig, ax = plt.subplots(1,2, figsize = (12,5))
            ax[0].set_title(f"IIR filter #{i + 1}")
            ax[0].plot(prev_time_points[start:end], np.gradient(sample/ max(sample)), label = "Corrected signal derivative")
            ax[0].plot(prev_time_points[start:end], sample/ max(sample), label = "Corrected signal derivative")
            ax[0].plot(prev_time_points[start:end], estimated_deltafunction, label = "Target signal derivative")
            ax[1].set_title(f"IIR correction #{i + 1}")
            ax[1].plot(prev_time_points, curr_correction/ max(np.abs(curr_correction)), 
                     color="C0", label = "Previous correction")
            ax[1].plot(time_points, corrected_pulse/ max(np.abs(corrected_pulse)),
                      color="C1", label = "Current correction")
            
            plt.legend()
            plt.show()
        curr_correction = corrected_pulse / max(corrected_pulse)

if SAVE_FILTERS:
    filters = {"iir_filters": iir_filters, "fir_filters": FIR_FILTERS}
    with open(FILTER_SAVE_LOCATION, "wb") as handle:
        pickle.dump(filters, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("--------- FILTERS GENERATED ----------")

if PLOT_PREDISTORTED_PULSE:
    print("--------- GENERATING PREDISTORTED PULSE ----------")

    _, predistorted_pulse = square_pulse(
        on_time=400, lpad=200, rpad=2000, dt=SAMPLING_PERIOD, amp=1
    )

    pulse_time_points = np.arange(len(predistorted_pulse)) * SAMPLING_PERIOD

    if DO_IIR:
        for i in range(len(iir_filters)):
            curr_filter = iir_filters[i]["correction"]
            _, predistorted_pulse = dlsim(
                curr_filter, predistorted_pulse, t=pulse_time_points
            )

    if DO_FIR:
        for i in range(FIR_FILTER_NUM):
            predistorted_pulse = np.squeeze(predistorted_pulse)
            filter_values = FIR_FILTERS[i]
            predistorted_pulse = convolve(
                predistorted_pulse, filter_values, mode="same"
            )

    predistorted_pulse_norm = predistorted_pulse / (
        max(abs(predistorted_pulse))
    )  # normalize pulse as OPX only allows values from 0-1

    plt.plot(
        pulse_time_points, predistorted_pulse_norm, label="fir filters", color="orange"
    )
    plt.title("Preview of a 2us predistorted square pulse")
    plt.show()
