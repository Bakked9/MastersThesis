# Import system modules
import arcpy
from arcpy import env

# Obtain a license for the ArcGIS 3D Analyst extension
arcpy.CheckOutExtension("3D")

# Set environment settings
env.workspace = "D:/New folder/IMG_2019"

try:
    # Create the list of IMG rasters
    rasterList = arcpy.ListRasters("*", "TIF")
    # Verify there are rasters in the list
    if rasterList:
        # Loop the process for each raster
        for raster in rasterList:
            # Set Local Variables
            outGeom = "POLYGON" # output geometry type
            # The [:-4] strips the .img from the raster name
            outPoly = "domain_" + raster[:-4] + ".shp"
            print "Creating footprint polygon for " + raster + "."
            #Execute RasterDomain
            arcpy.RasterDomain_3d(raster, outPoly, outGeom)
        print "Finished."
    else:
        "There are no TIF files in the " + env.workspace + " directory."
        
except Exception as e:
    # Returns any other error messages
    print e.message

Environment
