library(sf)
library(dplyr)
library(tmap)
library(viridis)

# -----------------------------
# Read data
# -----------------------------
setwd("C:/Users/nahomw/Desktop/from_mac/nahomworku/Desktop/uthealth/gra_project/synthetic-population/src/validate/geo_files/")

shp <- st_read("tl_2025_us_county.shp")
df  <- read.csv("../APR_MDC_map_1.csv")

# -----------------------------
# Prepare shapefile
# -----------------------------
shp$COUNTYFP <- as.numeric(shp$COUNTYFP)

# Texas only
shp_tx <- shp %>% filter(STATEFP == "48")

# -----------------------------
# Join JS data
# -----------------------------
map_df <- shp_tx %>%
  inner_join(df, by = c("COUNTYFP" = "PAT_COUNTY"))

# -----------------------------
# Static mode (publication)
# -----------------------------
tmap_mode("plot")

# -----------------------------
# Map
# -----------------------------
tm_shape(map_df) +
  
  # Choropleth: JS similarity
  tm_polygons(
    col = "JS_similarity",
    palette = "viridis",
    style = "cont",
    title = "JS Similarity (%)",
    border.col = "gray80",
    lwd = 0.25
  ) +
  
  # Bubble overlay: percentage (scaled to fit counties)
  tm_bubbles(
    size = "PERCENTAGE",
    scale = 0.8,              # 🔑 critical: keeps bubbles inside counties
    col = "black",
    alpha = 0.35,
    border.col = "black",
    border.alpha = 0.6,
    legend.size.show = FALSE  # 🔑 remove size legend
  ) +
  
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE,
    legend.outside.position = "right",
    legend.text.size = 0.8,
    legend.title.size = 0.9,
    main.title = "JS Similarity by County (APR_MDC)",
    main.title.size = 1
  )
