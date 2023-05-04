# Some notes

for rendering scenes:
- we want to have each round be accessible
- so, have as properties:
  - round names
  - dask arrays, accessible as channels: `scene.image_data[round_name] == list[Channel]`
  - napari layers, accessible as `scene.napari_layers[round_name][c]`
