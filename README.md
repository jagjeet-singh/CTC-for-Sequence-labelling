.png
# CTC-for-Sequence-labelling

Conventionally, sequence labelling is performed on sequences where each label can be isolated. However, in practical scenarios, this might not be the case. For example, in free hand writing, the letters often overlap and it is not possible to create bounding boxes for each letter. CTC considers the entire sequence as a whole and uses forward-backward algorithm to provide the appropriate labeliing.

![Alt text](assump1.png?raw=true "Example input")
