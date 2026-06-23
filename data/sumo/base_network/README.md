# SUMO Base Network Assets

Put the thesis SUMO network and detector files in this directory before running real SUMO experiments.

Expected default files:

- `test1.net.xml`: base merge-area SUMO network.
- `E2_info.xml`: lane-area detector additional file.

The default configs point to these paths:

```yaml
environment:
  net_file: data/sumo/base_network/test1.net.xml
  additional_file: data/sumo/base_network/E2_info.xml
```

Route files and generated `.sumocfg` files are written to `data/sumo/generated_routes/` at runtime and are intentionally not tracked.
