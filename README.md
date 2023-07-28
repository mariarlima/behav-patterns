# Discovering Behavioural Patterns Using Conversational Technology
This repository contains the code for data analysis and publishing figures of the [paper](https://ieeexplore.ieee.org/document/10168160) _Discovering Behavioural Patterns Using Conversational Technology for in-Home Health and Well-being Monitoring_. 

## Aim
- This study aims to investigate the integration of conversational agents in smart environments. 
- We fuse in-home activity data and voice interactions with conversational technology (Amazon Alexa) to analyse behaviour in households with people living with dementia (PLWD). 

## Data overview
- This study fuses in-home activity data and Alexa data to analyse behaviour in households with PLWD.
- Each household included a range of Internet of Things (IoT) technologies and remote health monitoring devices: passive infrared sensors (PIRs); door sensors; a sleeping
ma; physiological devices, including a pulse oximeter, scale, thermometer, and blood pressure cuff; and the smart speaker Amazon Echo Show.
- The activity data was extracted from [DCARTE](https://github.com/esoreq/dcarte).
- Alexa data consisted of text utterances and the corresponding timestamp.

To install DCARTE:
```
 $ pip install -U dcarte
```

### Datasets
All datasets used in this study are from the household subset. IDs were given to participant households to protect privacy. For the same reason, the content of Alexa interactions is not provided in the datasets shared in this repository.

- _df_activity.pkl_: aggregated in-home activity data.<sup>[*]</sup>
- _df_alexa.pkl_: Alexa data for investigating the prevalence of interactions and topics of interest beyond the novelty phase.
- _df_topic_embed.pkl_: Alexa data used for topic modelling with embeddings for each utterance (768-dimensional vector).  

<sub><sup>[*]</sup>Note: due to file size limitations, a compressed .pkl file is included in this repository</sub>.

## Analysis Overview
Broadly, our analysis inspects: 

1. The use of Alexa in households with PLWD over time, particularly to assess compliance with a daily well-being questionnaire and prevalence of topics of interest beyond the novelty phase.

2. Activity sequences in the 10-min period preceding or following interactions with Alexa to identify behavioural patterns, changes in those patterns, and the corresponding time periods.

3. Alexa usage in the week after health events occurred (information logged by a monitoring team, as detailed in Section III of the paper).

## Citing 

If you use or refer to the infrastructure, methods or data flow introduced please cite the [original paper](https://ieeexplore.ieee.org/document/10168160):

<sub>Lima, M.R., Su, T., Jouaiti, M., Wairagkar, M., Malhotra, P., Soreq, E., Barnaghi, P. and Vaidyanathan, R., 2023. Discovering Behavioural Patterns Using Conversational Technology for in-Home Health and Well-being Monitoring. _IEEE Internet of Things Journal_.</sub>

## License
This work is licensed under a [Creative Commons Attribution 4.0 License]( https://creativecommons.org/licenses/by/4.0/). 

Copyright © 2023 IEEE. Personal use of this material is permitted. However, permission to use this material for any other purposes must be obtained from the IEEE by sending a request to
pubs-permissions@ieee.org.

## Acknowledgment
This work is supported by the UK Dementia Research Institute (UKDRI7003) which receives its funding from UK DRI Ltd, funded by the UK Medical Research Council, Alzheimer’s Society and Alzheimer’s Research UK. This work is also funded by the Research England Grand Challenge Research Fund (GCRF) through Imperial College London, the Imperial Biomedical Research Centre, and the Imperial College London’s President’s PhD Scholarships. 

## Contact
Please contact mr3418@ic.ac.uk for any clarification or interest in the work. 
