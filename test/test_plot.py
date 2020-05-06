import pytest
import camels
import matplotlib.pyplot as plt

def test_plot():
    plt.ioff()
    gauge_id = 13331500
    camels.plot_basin(gauge_id)

