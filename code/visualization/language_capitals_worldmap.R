# World map of languages localized to associated national capitals.
#
# Required packages:
# install.packages(c("ggplot2", "sf", "rnaturalearth", "rnaturalearthdata", "ggrepel"))

required_packages <- c("ggplot2", "sf", "rnaturalearth", "rnaturalearthdata", "ggrepel")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  stop(
    "Please install missing package(s): ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

library(ggplot2)
library(sf)
library(rnaturalearth)
library(ggrepel)

script_file <- normalizePath(
  sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)),
  mustWork = FALSE
)

git_root <- system("git rev-parse --show-toplevel", intern = TRUE)
output_dir <- file.path(git_root, "figures/main")

languages <- data.frame(
  code = c(
    "ALB", "BEN", "BRP", "DUT", "FAR", "FRE", "GER", "HIN",
    "JAP", "KOR", "MAN", "ROM", "RUS", "SPA", "SOM", "TUR",
    "ENG"
  ),
  language = c(
    "Albanian", "Bengali", "Brazilian Portuguese", "Dutch", "Farsi",
    "French", "German", "Hindi", "Japanese", "Korean", "Mandarin Chinese",
    "Romanian", "Russian", "Spanish (Mexican)", "Somali", "Turkish",
    "English (North A.)"
  ),
  country = c(
    "Albania", "Bangladesh", "Brazil", "Netherlands", "Iran",
    "France", "Germany", "India", "Japan", "South Korea", "China",
    "Romania", "Russia", "Mexico", "Somalia", "Turkey",
    "United States"
  ),
  capital = c(
    "Tirana", "Dhaka", "Brasilia", "Amsterdam", "Tehran",
    "Paris", "Berlin", "New Delhi", "Tokyo", "Seoul", "Beijing",
    "Bucharest", "Moscow", "Mexico City", "Mogadishu", "Ankara",
    "Washington, DC"
  ),
  lon = c(
    19.8189, 90.4125, -47.8825, 4.9041, 51.3890,
    2.3522, 13.4050, 77.2090, 139.6917, 126.9780, 116.4074,
    26.1025, 37.6173, -99.1332, 45.3182, 32.8597,
    -77.0369
  ),
  lat = c(
    41.3275, 23.8103, -15.7942, 52.3676, 35.6892,
    48.8566, 52.5200, 28.6139, 35.6895, 37.5665, 39.9042,
    44.4268, 55.7558, 19.4326, 2.0469, 39.9334,
    38.9072
  ),
  family = c(
    "Albanian", "Indo-Aryan", "Romance", "Germanic", "Iranian",
    "Romance", "Germanic", "Indo-Aryan", "Japonic", "Koreanic",
    "Sino-Tibetan", "Romance", "Slavic", "Romance", "Afro-Asiatic",
    "Turkic", "Germanic"
  ),
  stringsAsFactors = FALSE
)

languages$label <- paste0(languages$code, " - ", languages$language)
english <- subset(languages, code == "ENG")
other_languages <- subset(languages, code != "ENG")

family_colors <- c(
  "Afro-Asiatic" = "#9A642F",
  "Albanian" = "#8F47A4",
  "Germanic" = "#245C8D",
  "Indo-Aryan" = "#C76500",
  "Iranian" = "#7464D8",
  "Japonic" = "#C21F30",
  "Koreanic" = "#008C89",
  "Romance" = "#087D5A",
  "Sino-Tibetan" = "#7A431F",
  "Slavic" = "#53671A",
  "Turkic" = "#9A3569"
)

family_outline_colors <- c(
  "Afro-Asiatic" = "#593817",
  "Albanian" = "#4B1D5D",
  "Germanic" = "#0E2F50",
  "Indo-Aryan" = "#693500",
  "Iranian" = "#3C3293",
  "Japonic" = "#6B0F1C",
  "Koreanic" = "#004A49",
  "Romance" = "#003F2D",
  "Sino-Tibetan" = "#3F210D",
  "Slavic" = "#2A3609",
  "Turkic" = "#561834"
)

world <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")

worldmap <- ggplot() +
  geom_sf(
    data = world,
    fill = "grey93",
    color = "white",
    linewidth = 0.25
  ) +
  geom_point(
    data = languages,
    aes(x = lon, y = lat),
    color = "#C43B2F",
    fill = "#F6C85F",
    shape = 21,
    size = 3.4,
    stroke = 0.7
  ) +
  ggrepel::geom_label_repel(
    data = other_languages,
    aes(x = lon, y = lat, label = label, fill = family, color = family),
    size = 6.4,
    label.size = 0.3,
    label.padding = unit(0.16, "lines"),
    box.padding = 0.45,
    point.padding = 0.25,
    min.segment.length = 0,
    segment.color = "grey45",
    max.overlaps = Inf,
    seed = 12
  ) +
  ggrepel::geom_label_repel(
    data = other_languages,
    aes(x = lon, y = lat, label = label),
    size = 6.4,
    color = "white",
    fill = NA,
    label.size = NA,
    label.padding = unit(0.16, "lines"),
    box.padding = 0.45,
    point.padding = 0.25,
    min.segment.length = 0,
    segment.color = NA,
    max.overlaps = Inf,
    seed = 12,
    show.legend = FALSE
  ) +
  ggrepel::geom_label_repel(
    data = english,
    aes(x = lon, y = lat, label = label, fill = family, color = family),
    size = 6.4,
    fontface = "bold",
    nudge_y = -7,
    label.size = 0.3,
    label.padding = unit(0.16, "lines"),
    box.padding = 0.45,
    point.padding = 0.25,
    min.segment.length = 0,
    segment.color = "grey45",
    max.overlaps = Inf,
    seed = 12
  ) +
  ggrepel::geom_label_repel(
    data = english,
    aes(x = lon, y = lat, label = label),
    size = 6.4,
    color = "white",
    fill = NA,
    fontface = "bold",
    nudge_y = -7,
    label.size = NA,
    label.padding = unit(0.16, "lines"),
    box.padding = 0.45,
    point.padding = 0.25,
    min.segment.length = 0,
    segment.color = NA,
    max.overlaps = Inf,
    seed = 12,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = family_colors, guide = "none") +
  scale_color_manual(values = family_outline_colors, guide = "none") +
  coord_sf(
    crs = "+proj=robin",
    default_crs = sf::st_crs(4326),
    xlim = c(-118, 152),
    ylim = c(-28, 62),
    expand = FALSE
  ) +
  labs(
    title = "First language backgrounds represented in the data"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major = element_line(color = "grey86", linewidth = 0.2),
    panel.background = element_rect(fill = "#F8FAFC", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "grey25"),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

print(worldmap)

ggsave(
  filename = file.path(output_dir, "language_capitals_worldmap.png"),
  plot = worldmap,
  width = 13,
  height = 6.4,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "language_capitals_worldmap.pdf"),
  plot = worldmap,
  width = 13,
  height = 6.4
)
