"""
This module contains the TreeMapConnection class, which is used to extract plot and tree data
from the TreeMapConnection raster using an ROI object.
"""

# Internal imports
from utils.raster import RasterConnection

# External imports
import numpy as np
from pandas import DataFrame
from dask import dataframe as dd
from shapely.geometry import Point
from geopandas import GeoDataFrame
from xarray import DataArray

# from rioxarray
from fastfuels_core.trees import TreeSample


class TreeMapConnection(RasterConnection):
    """
    Creates a RasterConnection with a connection type of "rioxarray" for
    TreeMap rasters. The URL for the raster is constructed from the version.
    """

    def __init__(self, treemap_path: str, tree_table_path: str, version: str, **kwargs):
        if version not in ["2014", "2016"]:
            raise ValueError(f"Invalid version: {version}")
        self.version = version
        self._tl_key = "tl_id" if self.version == "2014" else "tm_id"
        self.raster_path = treemap_path
        super().__init__(self.raster_path, connection_type="rioxarray", **kwargs)
        self.table_path = tree_table_path

        try:
            if self.table_path.endswith(".csv"):
                self._tree_table = dd.read_csv(self.table_path)
            elif self.table_path.endswith(".parquet"):
                self._tree_table = dd.read_parquet(self.table_path)
            else:
                raise NotImplementedError(
                    f"Unsupported file format for tree table: {self.table_path}"
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Invalid tree table path: {self.table_path}")

    @staticmethod
    def get_plots_dataframe_from_raster(raster: DataArray) -> GeoDataFrame:
        """
        Extracts the plot data from the raster and returns it as a GeoDataFrame.

        Parameters
        ----------
        raster: DataArray
            The raster data containing the plot data.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the plot data.
        """
        # Create a grid of coordinates
        x_coords = raster.coords["x"].values
        y_coords = raster.coords["y"].values
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Create a GeoDataFrame for pixel centers
        plots_gdf = GeoDataFrame(
            {
                "PLOT_ID": raster.values.ravel(),
            },
            geometry=[Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())],
            crs=raster.rio.crs,
        )

        return plots_gdf

    def query_trees_by_plots(self, plots: DataFrame | GeoDataFrame) -> TreeSample:
        plot_ids = plots["PLOT_ID"].unique().tolist()
        trees_delayed = self._tree_table[self._tree_table[self._tl_key].isin(plot_ids)]
        trees_delayed = trees_delayed.reset_index().rename(columns={"index": "TREE_ID"})
        trees_delayed = trees_delayed[
            [
                "TREE_ID",
                self._tl_key,
                "SPCD",
                "STATUSCD",
                "DIA",
                "HT",
                "ACTUALHT",
                "CR",
                "TPA_UNADJ",
            ]
        ]
        trees_delayed["HT"] = trees_delayed["ACTUALHT"].where(
            trees_delayed["ACTUALHT"].notnull(), trees_delayed["HT"]
        )
        trees_delayed = trees_delayed.drop(columns=["ACTUALHT"])
        trees = trees_delayed.compute()
        trees = trees.rename(columns={self._tl_key: "PLOT_ID"})

        return convert_treemap_data_to_fastfuels(trees)


def convert_treemap_data_to_fastfuels(data: DataFrame) -> TreeSample:
    """


    Parameters
    ----------
    data: DataFrame
        A dataframe containing TreeMapConnection data in imperial units.

    Returns
    -------
    TreeCollection
        A TreeCollection object containing the TreeMapConnection data
        converted to metric units and renamed to match the required columns
        of the TreeCollection object.
    """
    # Create a copy of the data
    metric_data = data.copy()

    # Convert CR from percentage to fraction
    metric_data["CR"] = metric_data["CR"] / 100

    # Convert HT from feet to meters
    metric_data["HT"] = metric_data["HT"] * 0.3048

    # Convert DIA from inches to centimeters
    metric_data["DIA"] = metric_data["DIA"] * 2.54

    # Convert TPA_UNADJ from trees per acre to trees per m^2
    metric_data["TPA_UNADJ"] = metric_data["TPA_UNADJ"] * 2.47105  # ac to ha
    metric_data["TPA_UNADJ"] = metric_data["TPA_UNADJ"] / 10000  # ha to m^2
    metric_data = metric_data.rename(columns={"TPA_UNADJ": "TPA"})

    return TreeSample(metric_data)
