from camels.data import gauge_information, gauge_ids
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_overview():
    """
    Plots overview over geographic locations of gauges available in
    the dataset.
    """
    try:
        from cartopy.io.img_tiles import Stamen
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        import cartopy.crs as ccrs

        tiler = Stamen("terrain")
        mercator = tiler.crs

        f = plt.figure(figsize = (10, 8))
        gs = GridSpec(1, 2, width_ratios=[1.0, 0.03])

        ax = plt.subplot(gs[0], projection=mercator)
        ax.set_extent([-130, -60, 20, 50])
        ax.add_image(tiler, 5)
        #ax.coastlines('10m', lw=0.5)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        xticks = np.arange(-135, -64, 10)
        yticks = np.arange(25, 46, 10)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        img=ax.scatter(gauge_information["longitude"],
                    gauge_information["latitude"],
                    transform=ccrs.PlateCarree(),
                    c=gauge_information["gauge id"],
                    s=2,
                    cmap="plasma")
        ax.set_title("Geographic locations of gauges in CAMELS dataset")
        asp = ax.get_aspect()

        ax = plt.subplot(gs[1])
        cb = plt.colorbar(img, label="Gauge ID", cax=ax)
        cb.formatter.set_useOffset(False)
        cb.formatter.set_scientific(False)
        cb.formatter.set_powerlimits((-10, 20))
        cb.update_ticks()
        plt.tight_layout()

    except:
        url = "http://spfrnd.de/datasets/camels/overview.png"
        img = plt.imread(url)
        plt.imshow(img)
        plt.gca().set_axis_off()

def plot_basin(gauge_id,
               dlat=0.05,
               dlon=0.1,
               tile_level=12):
    """
    Plots location of gauge and surrounding territory.

    Args:
        gauge_id(int): The ID of the gauge to display
        dlat: Meridional extent of the map in degrees.
        dlon: Equatorial extent of the map in degrees.
        tile_level: The resolution level to use for the terrain
            in the background.
    """
    if not gauge_id in gauge_ids:
        other = gauge_ids[np.argmin(np.abs(int(gauge_id) - gauge_ids))]
        raise ValueError("Gauge ID {} is not available from this dataset. The "
                        "closest ID available is {}.".format(gauge_id, other))

    try:
        from cartopy.io.img_tiles import Stamen
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        import cartopy.crs as ccrs

        lat = float(gauge_information.loc[gauge_information["gauge id"] == gauge_id]["latitude"])
        lon = float(gauge_information.loc[gauge_information["gauge id"] == gauge_id]["longitude"])

        tiler = Stamen("terrain")
        mercator = tiler.crs

        f = plt.figure(figsize = (8, 6))
        gs = GridSpec(1, 1)

        ax = plt.subplot(gs[0], projection=mercator)

        lon_min = lon - dlon
        lon_max = lon + dlon
        lat_min = lat - dlat
        lat_max = lat + dlat
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        ax.add_image(tiler, tile_level)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        lon_min_r = np.round(lon - dlon, decimals=1)
        lon_max_r = np.round(lon + dlon, decimals=1)
        lat_min_r = np.round(lat - dlat, decimals=1)
        lat_max_r = np.round(lat + dlat, decimals=1)

        xticks = np.arange(lon_min, lon_max, 0.05)
        yticks = np.arange(lat_min, lat_max, 0.05)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        img=ax.scatter([lon], [lat], transform=ccrs.PlateCarree(),
                    c="k", marker="x", s=50, label="Gauge location")
        ax.legend()
    except:
        url = "http://spfrnd.de/datasets/camels/plots/{}.png".format(gauge_id)
        img = plt.imread(url)
        plt.imshow(img)
        plt.gca().set_axis_off()
