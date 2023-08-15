import pickle

FILTER_PATHS = ["filters/pi_scope_filter_20230813_IIR_4.pickle", "filters/pi_scope_filter_20230811_IIR.pickle"]

SAVE_PATH = "pi_scope_filter_20230813_IIR.pickle"

iir_filters = []
fir_filters = []
for i in range(len(FILTER_PATHS)):
    with open(FILTER_PATHS[i], "rb") as filter_file:
        curr_filters = pickle.load(filter_file)
        iir_filters.extend(curr_filters["iir_filters"])
        fir_filters.extend(curr_filters["fir_filters"])
print(len(fir_filters))
print(len(iir_filters))

filters = {"iir_filters": iir_filters, "fir_filters": fir_filters}
with open(SAVE_PATH, "wb") as output_file:
    pickle.dump(filters, output_file, protocol=pickle.HIGHEST_PROTOCOL)
