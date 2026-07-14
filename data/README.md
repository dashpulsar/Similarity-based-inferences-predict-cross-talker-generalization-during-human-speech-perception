# Processed data
The file *-behavioral-data.csv files contain the preprocessed exposure and test data for all data sets we use. Unlike the raw data, these data files have standardized variable names, and contain all and only the information relevant for our computational modeling. Each row corresponds to one observation---i.e., the data are stored at the highest resolution that was available for that data set.

  * experiment_id -- a unique ID identifying the experiment: 
    * AN19 - Alexander & Nygaard (2019, Experiment 2)
    * B23 - Bradlow, Bassard, & Paller (2023)
    * X21 - Xie, Liu, & Jaeger (2021, Experiment 1a)
  * participant_id -- a unique ID across all studies that identifies a listener who completed the experiment. Format: {experiment_id}.number
  * exposure_test_condition_id: a unique id identifying the condition indicated in the data table.
  * exposure_test_condition.original: the condition name for this exposure or test trial that was used in the original article that published the dataset.
  * exposure_condition.accent: control (no exposure or L1-English exposure), single (when only one accent is experienced during exposure), multi (when multiple accents are experience during exposure)
  * exposure_condition.talker: none, single (when only one accent is experienced during exposure), multi (when multiple accents are experience during exposure)
  * test_condition.accent_generalization: control, cross-accent (when current test item assesses generalization across accents), within-accent (when current test item assesses generalization within accents). This variable is only defined for test trials.
  * test_condition.talker_generalization: none (no talkers during exposure), cross-talker (when current test item assesses generalization across talkers), within-talker (when current test item assesses generalization within talkers). This variable is only defined for test trials.
  * exposure_accents, test_accents: alphabetically sorted list of of unique three-letter accent identifiers also used in data table. These variables store all accents experienced by the participant during exposure/test. These variables store all talkers experienced by the participant during exposure/test. The format is {experimentID}.{accent_id}{talker_id}
  * exposure_talkers, test_talkers: alphabetically sorted list of unique talker identifiers.
  * phase: exposure or test
  * trial: from the start of the experiment, which trial is this? (1, ...)
  * trial.within_phase: from start of the phase, which trial is this? (1, ...)
  * item_id: a unique id identifying a speech recording. The format is {experiment_id}.{accent_id}.{talker_id}.{item_id}
    * For AN19: {item_id} has the format W{number}, indicating the isolated word recording. 
    * For B23: {item_id} has the format S{number}, indicating the sentence recording
    * For X21: {item_id} has the format S{number}.W{number}, indicating which sentence the word is in, and which keyword in that sentence it is.
  * item_unit: word or sentence, indicating whether items are stored at the word-level (AN19, X21) or sentence-level (B23).
  * item_accent: unique three-letter accent identifiers for the current item.
  * item_talker: a unique talker identifier for the current talker.
  * response_expected, response_expected.ipa: the correct, experimenter-intended, transcript of the item---either as transcript or in using the international phonetic alphabet (IPA).
  * response_provided, response_provided.ipa: the participant-provided transcript of the item. 
  * response_correct, response_incorrect: the count of correct/incorrect responses for this item (0 or 1 for AN19, X21; a count from 0 to more for B23 since those data are stored at the sentence-level but scored in terms of the correctly answered keywords in that sentence).

# Raw data 
The raw_data/ folder contain the behavioral measurements and the sounds stimuli (and were available Praat textgrids with annotated sentence, word or segment boundaries) from the published data sets we analyze. 

The folders also contain the NIH-posted preprints of the papers that describe the experiments from which these data were collected.