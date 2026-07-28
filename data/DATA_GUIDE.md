# Data Preparation Guide

Raw downloaded data is not committed to this repository.

Create the following local structure before running
`notebooks/01_site_selection.ipynb`:

```text
data/
├── DATA_GUIDE.md
├── raw/
│   ├── census_da_boundaries.geojson
│   ├── census_da_attributes.csv
│   ├── traffic_counts.geojson
│   ├── transit_gtfs.zip
│   ├── public_destinations.geojson
│   └── additional_public_chargers.geojson   # optional
└── processed/
```

## Data downloaded automatically by the notebook

The main notebook retrieves these datasets through the City of Toronto
CKAN API:

### City-operated EV charging stations

Dataset page:

```text
https://open.toronto.ca/dataset/city-operated-electric-vehicle-charging-station-map/
```

CKAN package ID:

```text
city-operated-electric-vehicle-charging-station-map
```

### Green P parking facilities

Dataset page:

```text
https://open.toronto.ca/dataset/green-p-parking/
```

CKAN package ID:

```text
green-p-parking
```

The notebook uses the `green-p-parking-2019` resource when available.

## Local source files

### 1. `census_da_boundaries.geojson`

Source:

- Statistics Canada, 2021 Census Dissemination Area Boundary Files
- https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21

Preparation:

1. Download the 2021 dissemination-area boundary file.
2. Read the downloaded Shapefile or File Geodatabase with GeoPandas.
3. retain Toronto dissemination areas;
4. preserve `DGUID` and `DAUID`;
5. export the result as:

```text
data/raw/census_da_boundaries.geojson
```

Example:

```python
import geopandas as gpd

boundaries = gpd.read_file(
    "path/to/downloaded_da_file.shp"
)

toronto_boundaries = boundaries[
    boundaries["CSDNAME"].eq("Toronto")
].copy()

toronto_boundaries.to_file(
    "data/raw/census_da_boundaries.geojson",
    driver="GeoJSON",
)
```

Column names in the original source may vary by format. Inspect the
downloaded metadata before filtering.

### 2. `census_da_attributes.csv`

Source:

- Statistics Canada, 2021 Census Profile Downloads
- https://www150.statcan.gc.ca/n1/en/catalogue/98-401-X

The site-selection notebook expects one row per dissemination area with:

| Column | Meaning |
|---|---|
| `DGUID` or `DAUID` | Census join key |
| `population` | total population |
| `households` | household count |
| `apartment_population` | prepared proxy for residents in apartments |
| `median_income` | median income measure used in the analysis |

Save the standardized result as:

```text
data/raw/census_da_attributes.csv
```

The `apartment_population` field is a prepared project variable rather
than a standard boundary-file attribute. Document the Census variables
and transformation used to derive it.

### 3. `traffic_counts.geojson`

Suggested source:

- City of Toronto Open Data, Traffic Volumes at Intersections for All
  Modes
- https://open.toronto.ca/dataset/traffic-volumes-at-intersections-for-all-modes/

Prepare one point or point-representative record per traffic-count
location. The required standardized numeric field is:

```text
traffic_volume
```

Save the result as:

```text
data/raw/traffic_counts.geojson
```

The notebook also recognizes `traffic_volume_per_hour` and
`total_vehicle`, but `traffic_volume` is preferred.

### 4. `transit_gtfs.zip`

Source:

- City of Toronto Open Data, TTC Routes and Schedules
- https://open.toronto.ca/dataset/ttc-routes-and-schedules/

Save the GTFS archive without extracting it:

```text
data/raw/transit_gtfs.zip
```

The archive must contain:

```text
stops.txt
```

Required stop fields:

- `stop_id`;
- `stop_lat`;
- `stop_lon`.

### 5. `public_destinations.geojson`

Suggested official sources:

- Parks and Recreation Facilities
- https://open.toronto.ca/dataset/parks-and-recreation-facilities/
- Library Branch General Information
- https://open.toronto.ca/dataset/library-branch-general-information/

Combine the relevant locations into one point layer. A simple schema is:

| Column | Meaning |
|---|---|
| `destination_id` | stable identifier |
| `destination_type` | `park`, `recreation`, or `library` |
| `name` | public location name |
| `geometry` | point geometry |

Save the combined layer as:

```text
data/raw/public_destinations.geojson
```

### 6. `additional_public_chargers.geojson` — optional

This optional file can contain additional licensed public-charger
locations not represented in the City-operated inventory.

Minimum requirement:

- valid point geometry.

Recommended fields:

- `charger_id`;
- `operator`;
- `source`;
- `geometry`.

When it is present, the notebook combines it with the City-operated
inventory and deduplicates nearby records.

## Files that may be committed

Raw files under `data/raw/` remain ignored.

After running the workflow, the following derived files may be committed
to support a public demo:

```text
data/processed/scored_ev_charging_candidates.geojson
data/processed/selected_ev_charging_sites.geojson
outputs/agent_candidate_summary.json
outputs/selected_site_sequence.csv
outputs/planning_configuration.json
outputs/scenario_results.csv
outputs/sensitivity_results.csv
```

Review source-data licences before publishing derived records.
