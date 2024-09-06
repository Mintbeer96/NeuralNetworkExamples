import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os

# Define the place name
'''
City Data Preprocessing
'''

# Define a function to create consistent bounding box
def get_bbox(center_lat, center_lon, km):
    # Convert km to degrees (approximate)
    lat_diff = km / 111  # 1 degree latitude ~ 111 km
    lon_diff = km / (111 * abs(np.cos(np.radians(center_lat))))  # Adjust by latitude
    north = center_lat + lat_diff / 2
    south = center_lat - lat_diff / 2
    east = center_lon + lon_diff / 2
    west = center_lon - lon_diff / 2
    return north, south, east, west

# Specify area size in kilometers
area_km = 5

# Define center coordinates for Manhattan and Florence
manhattan_center = (40.7831, -73.9712)  # Approx. center of Manhattan
dc_center = (38.90511, -77.03637)  # Approx. center of downtown Washington, D.C.
florence_center = (43.7696, 11.2558)  # Approx. center of Florence

place_name = "Manhattan, New York, USA"

# Download the boundary for the area
gdf = ox.geocode_to_gdf(place_name)

# bbox = gdf.total_bounds
# north, south, east, west = 40.900, 40.800, -73.900, -74.000
# # Create a polygon from the bounding box
# Get bounding boxes for both cities
manhattan_bbox = get_bbox(dc_center[0], dc_center[1], area_km)
manhattan_polygon = ox.utils_geo.bbox_to_poly(*manhattan_bbox)
# polygon = ox.utils_geo.bbox_to_poly(north, south, east, west)

# Download data within the bounding box (for example, buildings)
gdf_buildings = ox.features_from_polygon(manhattan_polygon, tags={'building': True})
gdf_roads = ox.features_from_polygon(manhattan_polygon, tags={'highway': True})
# gdf_rivers = ox.features_from_polygon(manhattan_polygon, tags={'waterway': 'river'})

# If gdf_roads contains other features like points or nodes, you can filter to only keep 'LineString' geometries (roads)
gdf_roads = gdf_roads[gdf_roads.geometry.type == 'LineString']
    # Set up the plot
fig, ax = plt.subplots(figsize=(20, 20))

# Plot the buildings
gdf_buildings.plot(ax=ax, facecolor='blue', edgecolor='red', linewidth=0.2, alpha=0.7)
gdf_roads.plot(ax=ax, facecolor='None', edgecolor='red', linewidth=0.2, alpha=0.7)
# gdf_rivers.plot(ax=ax, facecolor='None', edgecolor='cyan', linewidth=0.2, alpha=0.7)

# Set plot limits to match the bounding box
ax.set_xlim(manhattan_bbox[3], manhattan_bbox[2])
ax.set_ylim(manhattan_bbox[1], manhattan_bbox[0])
# # Plot the graph and save it at high resolution

# Set the background color to white
ax.set_facecolor('white')

# Hide the axis
ax.axis('off')

# Save the image at a high resolution
fig.savefig("buildings_highways_map.png", bbox_inches='tight', pad_inches=0, dpi=300)

# Close the figure to free memory
plt.close(fig)

#Use a more squared shape city
place_name = "Florence, Italy"

# Download the boundary for the area
gdf = ox.geocode_to_gdf(place_name)

bbox = gdf.total_bounds
# # Define a smaller bounding box manually for a focused area
# north, south, east, west = 43.865, 43.765, 11.260, 11.160
# Create a polygon from the bounding box
florence_bbox = get_bbox(florence_center[0], florence_center[1], area_km)
florence_polygon = ox.utils_geo.bbox_to_poly(*florence_bbox)

# Download data within the bounding box (for example, buildings)
gdf_buildings = ox.features_from_polygon(florence_polygon, tags={'building': True})
gdf_roads = ox.features_from_polygon(florence_polygon, tags={'highway': True})
# gdf_rivers = ox.features_from_polygon(florence_polygon, tags={'waterway': 'river'})

# If gdf_roads contains other features like points or nodes, you can filter to only keep 'LineString' geometries (roads)
gdf_roads = gdf_roads[gdf_roads.geometry.type == 'LineString']
    # Set up the plot
fig, ax = plt.subplots(figsize=(20, 20))

# Plot the buildings
gdf_buildings.plot(ax=ax, facecolor='blue', edgecolor='red', linewidth=0.2, alpha=0.7)
gdf_roads.plot(ax=ax, facecolor='None', edgecolor='red', linewidth=0.2, alpha=0.7)
# gdf_rivers.plot(ax=ax, facecolor='None', edgecolor='cyan', linewidth=0.2, alpha=0.7)

# Set plot limits to match the bounding box
ax.set_xlim(florence_bbox[3], florence_bbox[2])
ax.set_ylim(florence_bbox[1], florence_bbox[0])
# # Plot the graph and save it at high resolution

# Set the background color to white
ax.set_facecolor('white')

# Hide the axis
ax.axis('off')

# Save the image at a high resolution
fig.savefig("r_buildings_highways_map.png", bbox_inches='tight', pad_inches=0, dpi=300)

# Close the figure to free memory
plt.close(fig)


'''
Load images
'''
# Define the size of the tiles (e.g., 28x28 pixels)
tile_size = 64

# Open the saved high-definition image
c_img = Image.open("buildings_highways_map.png")

# Convert the image to RGB (optional, if not already in RGB)
c_img = c_img.convert("RGB")

# Get the dimensions of the original image
c_img_width, c_img_height = c_img.size

# Calculate the number of tiles in each dimension
c_num_tiles_x = c_img_width // tile_size
c_num_tiles_y = c_img_height // tile_size

# Chop the image into tiles
c_tiles = []

print("City Chopping......")
for i in range(c_num_tiles_x):
    for j in range(c_num_tiles_y):
        left = i * tile_size
        upper = j * tile_size
        right = left + tile_size
        lower = upper + tile_size
        tile = c_img.crop((left, upper, right, lower))
        c_tiles.append(tile)


# Open the saved high-definition image
r_img = Image.open("r_buildings_highways_map.png")

# Convert the image to RGB (optional, if not already in RGB)
r_img = r_img.convert("RGB")

# Get the dimensions of the original image
r_img_width, r_img_height = r_img.size

# Calculate the number of tiles in each dimension
r_num_tiles_x = r_img_width // tile_size
r_num_tiles_y = r_img_height // tile_size


r_tiles = []

print("Rural Chopping......")
for i in range(r_num_tiles_x):
    for j in range(r_num_tiles_y):
        left = i * tile_size
        upper = j * tile_size
        right = left + tile_size
        lower = upper + tile_size
        tile = r_img.crop((left, upper, right, lower))
        r_tiles.append(tile)

'''
OUTPUT
'''

city_folder = "city"
rural_folder = "rural"

city_dir = os.path.join("tile_test" , city_folder)
rural_dir = os.path.join("tile_test" , rural_folder)

import shutil
files = glob.glob("./tile_test/*")

for f in files:
    if os.path.isdir(f):
        shutil.rmtree(f)
    else:
        os.remove(f)


if not os.path.exists(city_dir):
    os.makedirs(city_dir)

if not os.path.exists(rural_dir):
    os.makedirs(rural_dir)

print("Saving......")
# Save the tiles (optional)
for idx, tile in enumerate(c_tiles):
    tile.save(city_dir + f"/tile_{idx}.png")

# Save the tiles (optional)
for idx, tile in enumerate(r_tiles):
    tile.save(rural_dir + f"/tile_{idx}.png")

print(f"Generated {len(c_tiles)} city tiles.")
print(f"Generated {len(r_tiles)} rural tiles.")
