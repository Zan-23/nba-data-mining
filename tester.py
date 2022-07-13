# import time
#
# from src.data_loader import load_data
#
# if __name__ == "__main__":
#     start_time = time.time()
#     print("Testing data_loader.py")
#
#     tmp_d = load_data()
#
#     print(tmp_d.head())
#     print("Testing data_loader.py took %s seconds" % (time.time() - start_time))
import cProfile
import io
import pstats

from src.data_loader import load_data

if __name__ == "__main__":
    profiler = cProfile.Profile()
    file_output = io.StringIO()

    profiler.enable()
    # run_simulation(False, False, False, directory="./../../data_dowloading/scenarios/scenario_real_whole_day")ž
    load_data()
    # print("sss")
    profiler.disable()

    # Arguments to stats: ncalls, cumtime
    stats = pstats.Stats(profiler, stream=file_output).sort_stats("cumtime")
    stats.print_stats()
    print(file_output.getvalue())
    # stats.dump_stats("./stats_file_1.dat")

    with open("./stats_file_v2.txt", 'w+') as f:
        f.write(file_output.getvalue())