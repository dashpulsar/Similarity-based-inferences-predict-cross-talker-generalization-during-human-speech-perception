# Heatmap of broad phonological similarity to English by first-language background.
#
# The scores below are a transparent, editable synthesis for visualization.
# 1.00 = most English-like for that feature; 0.00 = least English-like.
# They should be treated as expected transfer-pressure indicators, not measured
# learner error rates.
#
# Public source basis:
# - PHOIBLE: segment inventories and distinctive features, https://phoible.org/
# - WALS: phonological typology, https://wals.info/
# - ASJP: optional surface lexical/phonetic distance, https://asjp.clld.org/
# - lang2vec / URIEL: optional typological feature vectors,
#   https://github.com/antonisa/lang2vec

required_packages <- c("ggplot2")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  stop(
    "Please install missing package(s): ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

library(ggplot2)


git_root <- system("git rev-parse --show-toplevel", intern = TRUE)
output_dir <- file.path(git_root, "figures/main")

language_order <- c(
  "DUT", "GER", "FRE", "SPA", "BRP", "ROM", "RUS", "ALB",
  "FAR", "HIN", "BEN", "MAN", "JAP", "KOR", "TUR", "SOM"
)

language_labels <- setNames(language_order, language_order)

feature_rows <- list()

add_feature <- function(group, feature, basis, scores) {
  missing_codes <- setdiff(language_order, names(scores))
  if (length(missing_codes) > 0) {
    stop("Missing score(s) for ", feature, ": ", paste(missing_codes, collapse = ", "), call. = FALSE)
  }

  row <- data.frame(
    group = group,
    feature = feature,
    basis = basis,
    language = language_order,
    score = as.numeric(scores[language_order]),
    stringsAsFactors = FALSE
  )

  feature_rows[[length(feature_rows) + 1]] <<- row
}

add_feature(
  "Segmental",
  "Consonant inventory",
  "Broad overlap with English consonant inventory",
  c(ENG = 1, DUT = .8, GER = .8, FRE = .7, SPA = .65, BRP = .7, ROM = .7,
    RUS = .75, ALB = .85, FAR = .65, HIN = .75, BEN = .7, MAN = .55, JAP = .45,
    KOR = .45, TUR = .65, SOM = .65)
)

add_feature(
  "Segmental",
  "Vowel inventory",
  "Broad overlap with English vowel inventory",
  c(ENG = 1, DUT = .8, GER = .85, FRE = .6, SPA = .3, BRP = .45, ROM = .45,
    RUS = .55, ALB = .5, FAR = .4, HIN = .5, BEN = .5, MAN = .45, JAP = .3,
    KOR = .45, TUR = .45, SOM = .55)
)

add_feature(
  "Segmental",
  "Fricative/affricate inventory",
  "Broad overlap with English fricatives and affricates",
  c(ENG = 1, DUT = .75, GER = .75, FRE = .7, SPA = .55, BRP = .65, ROM = .7,
    RUS = .75, ALB = .85, FAR = .55, HIN = .7, BEN = .65, MAN = .55, JAP = .4,
    KOR = .35, TUR = .6, SOM = .6)
)

add_feature(
  "Segmental",
  "Voicing contrasts",
  "PHOIBLE/WALS-informed inventory contrast",
  c(ENG = 1, DUT = .9, GER = .85, FRE = .9, SPA = .8, BRP = .8, ROM = .85,
    RUS = .8, ALB = .9, FAR = .8, HIN = .85, BEN = .8, MAN = .35, JAP = .55,
    KOR = .35, TUR = .75, SOM = .75)
)

add_feature(
  "Segmental",
  "Aspiration contrast",
  "Contrastive or strongly conditioned aspiration relative to English",
  c(ENG = 1, DUT = .45, GER = .75, FRE = .25, SPA = .2, BRP = .2, ROM = .25,
    RUS = .25, ALB = .3, FAR = .25, HIN = .75, BEN = .75, MAN = .75, JAP = .25,
    KOR = .65, TUR = .25, SOM = .35)
)

# add_feature(
#   "Segmental",
#   "Rhotic",
#   "Similarity to North American English /r/ realization",
#   c(ENG = 1, DUT = .35, GER = .3, FRE = .25, SPA = .25, BRP = .25, ROM = .35,
#     RUS = .25, ALB = .35, FAR = .35, HIN = .45, BEN = .35, MAN = .45, JAP = .15,
#     KOR = .2, TUR = .35, SOM = .35)
# )

add_feature(
  "Segmental",
  "Interdental fricatives",
  "Availability of sounds like thin/this",
  c(ENG = 1, DUT = 0, GER = 0, FRE = 0, SPA = .1, BRP = 0, ROM = 0,
    RUS = 0, ALB = .9, FAR = 0, HIN = .1, BEN = .1, MAN = 0, JAP = 0,
    KOR = 0, TUR = 0, SOM = .15)
)

add_feature(
  "Segmental",
  "Tense/lax vowels",
  "Approximate coverage of English vowel quality contrasts",
  c(ENG = 1, DUT = .8, GER = .85, FRE = .45, SPA = .25, BRP = .35, ROM = .35,
    RUS = .45, ALB = .45, FAR = .35, HIN = .45, BEN = .4, MAN = .35, JAP = .25,
    KOR = .35, TUR = .35, SOM = .45)
)

add_feature(
  "Segmental",
  "v/w distinction",
  "Separate treatment of English /v/ and /w/",
  c(ENG = 1, DUT = .85, GER = .5, FRE = .65, SPA = .3, BRP = .6, ROM = .6,
    RUS = .45, ALB = .7, FAR = .45, HIN = .35, BEN = .3, MAN = .35, JAP = .15,
    KOR = .15, TUR = .4, SOM = .45)
)

add_feature(
  "Segmental",
  "l/r distinction",
  "Separate treatment of English /l/ and /r/",
  c(ENG = 1, DUT = .75, GER = .75, FRE = .7, SPA = .7, BRP = .65, ROM = .75,
    RUS = .75, ALB = .75, FAR = .75, HIN = .75, BEN = .65, MAN = .75, JAP = .1,
    KOR = .2, TUR = .75, SOM = .75)
)

add_feature(
  "Phonotactics",
  "Syllable complexity",
  "Overall similarity to English syllable-shape range",
  c(ENG = 1, DUT = .85, GER = .85, FRE = .65, SPA = .45, BRP = .5, ROM = .65,
    RUS = .85, ALB = .75, FAR = .45, HIN = .55, BEN = .45, MAN = .15, JAP = .1,
    KOR = .2, TUR = .4, SOM = .45)
)

add_feature(
  "Phonotactics",
  "Complex onset clusters",
  "Compatibility with English word-initial clusters",
  c(ENG = 1, DUT = .9, GER = .9, FRE = .75, SPA = .45, BRP = .45, ROM = .75,
    RUS = .95, ALB = .85, FAR = .15, HIN = .55, BEN = .45, MAN = .1, JAP = .05,
    KOR = .1, TUR = .25, SOM = .35)
)

add_feature(
  "Phonotactics",
  "Complex coda clusters",
  "Compatibility with English word-final clusters",
  c(ENG = 1, DUT = .85, GER = .85, FRE = .55, SPA = .35, BRP = .35, ROM = .55,
    RUS = .85, ALB = .75, FAR = .55, HIN = .45, BEN = .35, MAN = .05, JAP = .05,
    KOR = .1, TUR = .35, SOM = .35)
)

add_feature(
  "Phonotactics",
  "Final voiced obstruents",
  "Compatibility with English final voicing contrasts",
  c(ENG = 1, DUT = .35, GER = .3, FRE = .65, SPA = .65, BRP = .55, ROM = .65,
    RUS = .3, ALB = .75, FAR = .7, HIN = .65, BEN = .55, MAN = .15, JAP = .1,
    KOR = .1, TUR = .45, SOM = .65)
)

add_feature(
  "Phonotactics",
  "Unstressed vowel reduction",
  "Similarity to English schwa/reduced-vowel patterns",
  c(ENG = 1, DUT = .65, GER = .65, FRE = .25, SPA = .15, BRP = .35, ROM = .35,
    RUS = .65, ALB = .35, FAR = .25, HIN = .25, BEN = .25, MAN = .15, JAP = .05,
    KOR = .1, TUR = .1, SOM = .2)
)

add_feature(
  "Supra-segmental",
  "Lexical tone",
  "Similarity to English absence of lexical tone",
  c(ENG = 1, DUT = 1, GER = 1, FRE = 1, SPA = 1, BRP = 1, ROM = 1,
    RUS = 1, ALB = 1, FAR = 1, HIN = 1, BEN = 1, MAN = .05, JAP = .6,
    KOR = .8, TUR = 1, SOM = .55)
)

add_feature(
  "Supra-segmental",
  "Lexical stress",
  "Presence and functional similarity of word stress",
  c(ENG = 1, DUT = .85, GER = .85, FRE = .25, SPA = .75, BRP = .75, ROM = .75,
    RUS = .8, ALB = .7, FAR = .45, HIN = .45, BEN = .35, MAN = .05, JAP = .25,
    KOR = .15, TUR = .45, SOM = .55)
)

add_feature(
  "Supra-segmental",
  "Variable stress placement",
  "Similarity to English non-fixed stress placement",
  c(ENG = 1, DUT = .75, GER = .75, FRE = .15, SPA = .65, BRP = .65, ROM = .65,
    RUS = .85, ALB = .65, FAR = .2, HIN = .4, BEN = .3, MAN = .05, JAP = .2,
    KOR = .1, TUR = .2, SOM = .35)
)

add_feature(
  "Supra-segmental",
  "Stress-timed rhythm",
  "Similarity to English stress-timed timing patterns",
  c(ENG = 1, DUT = .8, GER = .8, FRE = .35, SPA = .25, BRP = .35, ROM = .35,
    RUS = .65, ALB = .45, FAR = .35, HIN = .25, BEN = .25, MAN = .15, JAP = .05,
    KOR = .15, TUR = .15, SOM = .25)
)

add_feature(
  "Supra-segmental",
  "Sentence intonation",
  "Broad similarity of phrase stress and pitch movement",
  c(ENG = 1, DUT = .75, GER = .75, FRE = .55, SPA = .6, BRP = .6, ROM = .6,
    RUS = .55, ALB = .55, FAR = .5, HIN = .45, BEN = .45, MAN = .25, JAP = .3,
    KOR = .35, TUR = .45, SOM = .4)
)


# add_feature(
#   "Phonotactics",
#   "Word-final consonants",
#   "Availability of English-like final consonants",
#   c(ENG = 1, DUT = .85, GER = .85, FRE = .55, SPA = .65, BRP = .55, ROM = .7,
#     RUS = .85, ALB = .8, FAR = .75, HIN = .65, BEN = .55, MAN = .2, JAP = .1,
#     KOR = .35, TUR = .7, SOM = .65)
# )


similarity_long <- do.call(rbind, feature_rows)

feature_levels <- unique(similarity_long$feature)
similarity_long$feature <- factor(similarity_long$feature, levels = rev(feature_levels))
similarity_long$group <- factor(
  similarity_long$group,
  levels = c("Segmental", "Phonotactics", "Supra-segmental")
)
similarity_long$language <- factor(similarity_long$language, levels = language_order)

wide_scores <- reshape(
  similarity_long,
  idvar = c("group", "feature", "basis"),
  timevar = "language",
  direction = "wide"
)
names(wide_scores) <- sub("^score\\.", "", names(wide_scores))

write.csv(
  wide_scores,
  file = file.path(output_dir, "english_phonology_similarity_scores.csv"),
  row.names = FALSE
)

heatmap <- ggplot(similarity_long, aes(x = language, y = feature, fill = score)) +
  geom_tile(color = "white", linewidth = 0.8) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  scale_x_discrete(labels = language_labels) +
  scale_fill_gradientn(
    colors = c("#7A1F1F", "#C76D3A", "#F1D6A6", "#7FB7A3", "#116A65"),
    limits = c(0, 1),
    breaks = c(0, .25, .5, .75, 1),
    labels = c("0", ".25", ".50", ".75", "1"),
    name = "Similarity\nto English"
  ) +
  labs(
    title = "Phonological similarity of L1s in the data to English"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    panel.spacing.y = unit(0.7, "lines"),
    strip.placement = "outside",
    strip.text.y.left = element_text(size = 12.5, angle = 270, face = "bold", hjust = 0.5),
    axis.title = element_blank(),
    axis.text.x = element_text(size = 14, color = "grey15"),
    axis.text.y = element_text(size = 9, angle = 22.5, hjust = 1, color = "grey15"),
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
  )

print(heatmap)

ggsave(
  filename = file.path(output_dir, "english_phonology_similarity_heatmap.png"),
  plot = heatmap,
  width = 11,
  height = 6,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "english_phonology_similarity_heatmap.pdf"),
  plot = heatmap,
  width = 11,
  height = 6
)
