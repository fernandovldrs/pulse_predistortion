import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.optimize import curve_fit


data_file_path = "./data/130407_cut.h5"
output_filename = "converted_pi_scope.npz"

CLOCK_PERIOD = 4e-9  # It is unlikely that this should ever be changed

PLOT_INDIVIDUAL_FITS = False
PLOT_FINAL_GRAPH = True


# Gaussian function for curve fitting
def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c


def gaussian_fit(y, x):
    mean_arg = np.argmax(y) #np.argmin(y)
    mean = x[mean_arg]
    sigma = 5e6  # np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    fit_range = int(0.2 * len(x))
    x_sample = x[max([mean_arg - fit_range,0]):min([mean_arg + fit_range, len(x)])]
    y_sample = y[max([mean_arg - fit_range,0]):min([mean_arg + fit_range, len(x)])]

    # popt, pcov = curve_fit(
    #     gaussian,
    #     x_sample,
    #     y_sample,
    #     bounds=(
    #         (-np.inf, min(x_sample), -np.inf, -np.inf),
    #         (0, max(x_sample), np.inf, np.inf),
    #     ),
    #     p0=[min(y_sample) - max(y_sample), mean, sigma, max(y_sample)],
    # )
    popt, pcov = curve_fit(
        gaussian,
        x_sample,
        y_sample,
        bounds=(
            (0, min(x_sample), -np.inf, -np.inf),
            (np.inf, max(x_sample), np.inf, np.inf),
        ),
        p0=[max(y_sample) - min(y_sample), mean, sigma, min(y_sample)],
    )
    return popt

def convert_flux_pi_plot(
    z_avg,
    qubit_freq,
    time_delay,
    plot_final: bool = False,
    plot_individual_fits: bool = False,
) -> np.ndarray:
    """
    To convert our pi pulse from clock cycles to ns, we use interpolation
    to fill in the missing points. While this creates additional data, it
    will not affect our results as interpolation will be done in dlsim if
    it is not already done here. Hence, we'll do the interpolation here to
    simplify things.
    """
    z_avg_t = np.transpose(z_avg)
    qubit_freq_t = np.transpose(qubit_freq)
    data_points = len(z_avg_t)
    fit_data = [0] * data_points
    print("---------- PERFORMING GAUSSIAN FIT ----------")
    for i in range(data_points):
        curr_opt = gaussian_fit(z_avg_t[i], qubit_freq_t[i])
        fit_data[i] = -1 * curr_opt[1]
        print(f"Fitting for point {i+1} out of {data_points}")
        print(curr_opt)
        if plot_individual_fits:
            plt.scatter(qubit_freq_t[i], z_avg_t[i], label="fit")
            plt.plot(
                qubit_freq_t[i], gaussian(qubit_freq_t[i], *curr_opt), label="data"
            )
            plt.show()
    print("---------------------- DONE ----------------------")

    print("---------- CONVERTING TIMESCALE FROM CLOCK CYCLES TO NS ----------")
    time_delay_ns = time_delay * CLOCK_PERIOD
    num_points = 4 * max(time_delay)
    full_timesteps = np.linspace(time_delay_ns[0], max(time_delay_ns), num_points)
    converted_data = np.interp(full_timesteps, time_delay_ns, fit_data)
    print("------------------------------ DONE ------------------------------")

    if plot_final:
        plt.scatter(full_timesteps, converted_data)
        plt.show()
    return converted_data


def run_convert_flux_pi_plot(
    file_path: str, plot_final: bool = False, plot_individual_fits: bool = False
) -> np.ndarray:
    data_file = h5py.File(file_path)
    z_avg = data_file["data"]
    qubit_freq = data_file["x"]
    time_delay = data_file["y"][0]

    converted_data = convert_flux_pi_plot(
        z_avg,
        qubit_freq,
        time_delay,
        plot_final=plot_final,
        plot_individual_fits=plot_individual_fits,
    )
    return converted_data


if __name__ == "__main__":
    converted_data = run_convert_flux_pi_plot(
        file_path=data_file_path,
        plot_final=PLOT_FINAL_GRAPH,
        plot_individual_fits=PLOT_INDIVIDUAL_FITS,
    )
    np.savez(output_filename, data=converted_data)
    print(f"Converted waveform saved to {os.getcwd()}/{output_filename}")
