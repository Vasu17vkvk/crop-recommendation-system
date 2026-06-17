import ee

SERVICE_ACCOUNT = None

credentials = ee.ServiceAccountCredentials(
    None,
    "service-account.json"
)

ee.Initialize(credentials)

point = ee.Geometry.Point([77.2090, 28.6139])

image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2024-01-01", "2024-12-31")
    .median()
)

ndvi = image.normalizedDifference(
    ["B8", "B4"]
).rename("NDVI")

value = ndvi.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=point,
    scale=10
)

print(value.getInfo())
