# Shree KRISHNya Namaha
# Some common utilities
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024


def start_matlab_engine():
    import matlab.engine

    print('Starting MatLab Engine')
    matlab_engine = matlab.engine.start_matlab()
    print('MatLab Engine active')
    return matlab_engine
