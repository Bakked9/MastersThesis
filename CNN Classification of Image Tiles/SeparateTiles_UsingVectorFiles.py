import os, sys, arcpy

AOI     = "Path to Training Sample Polygon.shp"  # Polygons with area of interest that overlap with image tiles
InFolder = "Path" # the folder containing the rasters to search from
OutFolder = "Path" # the folder to copy to

arcpy.env.overwriteOutput =True
#Get the extent of each polygon feature individually from AOI, in a list
listExt = []
cursor = arcpy.SearchCursor(AOI)
for row in cursor:
    ext_row = row.getValue('Shape').extent  #NOTE:Geometry field name must match shapefile table   
    listExt.append(ext_row)
del cursor
arcpy.env.workspace = InFolder

for ThisRas in arcpy.ListRasters():
    rDesc = arcpy.Describe(ThisRas)
    rExt  = rDesc.extent
    for sExt in listExt:
    # check if this extent is related spatially
    # by using not disjoint
        if sExt.disjoint(rExt):
            arcpy.AddMessage("Raster %s is outside" % (ThisRas))
        else:
            arcpy.AddMessage("Raster %s overlaps" % (ThisRas))
            outFile = os.path.join(OutFolder,ThisRas)
            arcpy.Copy_management(os.path.join(InFolder,ThisRas),outFile)
